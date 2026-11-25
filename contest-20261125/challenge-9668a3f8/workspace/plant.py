"""T0 configuration adapter. All plasma evolution uses maintained Gym-TORAX/TORAX."""
from __future__ import annotations

import copy
import dataclasses
import functools
import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from torax._src import math_utils
from torax._src.internal_boundary_conditions import internal_boundary_conditions as ibc
from gymtorax.action_handler import EcrhAction, GenericHeatAction
from gymtorax.envs.base_env import BaseEnv
from gymtorax.observation_handler import AllObservation
from torax._src.state import SimError
from torax._src.transport_model import qlknn_transport_model


def numeric_keys(value):
    """JSON encodes the native interpolation's numeric dictionary keys as strings."""
    if isinstance(value, list):
        return [numeric_keys(x) for x in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        try:
            converted = float(key)
        except (TypeError, ValueError):
            converted = key
        result[converted] = numeric_keys(item)
    return result


def scenario_config(directory: Path, scenario: dict, level: str, horizon=None):
    case = json.loads((directory / 'case.json').read_text())
    config = numeric_keys(json.loads((directory / 'torax-config.json').read_text()))
    cfg = copy.deepcopy(config)
    settings = case['numerics'][level]
    cfg['geometry'].update(n_rho=settings['n_rho'], geometry_directory=str(directory))
    cfg['numerics'].update(
        fixed_dt=settings['fixed_dt_s'],
        t_final=case['horizon_s'] if horizon is None else horizon,
    )
    start = scenario['event_start_s']
    stop = start + case['scenario_distribution']['event_ramp_s']
    cfg['plasma_composition']['Z_eff'] = {
        0.0: {0.0: 1.6, 1.0: 1.6},
        start: {0.0: 1.6, 1.0: 1.6},
        stop: {0.0: scenario['Z_eff_after_event'], 1.0: scenario['Z_eff_after_event']},
    }
    for name in ('T_i', 'T_e'):
        for rho in cfg['profile_conditions'][name][0.0]:
            cfg['profile_conditions'][name][0.0][rho] *= scenario['initial_temperature_multiplier']
    return case, cfg


class CoreHeatEnv(BaseEnv):
    """Only defines native configuration/actions; it does not implement a plasma model."""

    def __init__(self, directory: Path, scenario: dict, level: str, *, horizon=None, store_history=False):
        self.case, self.native_config = scenario_config(directory, scenario, level, horizon)
        self.core_domain = json.loads((directory / 'qlknn-core-domain.json').read_text())['ranges']
        self.native_dt = self.case['numerics'][level]['fixed_dt_s']
        self.previous_power = np.array(self.case['actuator']['initial_MW'], dtype=float)
        super().__init__(log_level='warning', store_history=store_history)

    def _get_torax_config(self):
        return {
            'config': self.native_config,
            'discretization': 'fixed',
            # Policy runs at 0.5 s; maintained actions run at native accepted
            # time boundaries so domain/safety checks can see each state.
            'ratio_a_sim': 1,
        }

    def _define_action_space(self):
        a = self.case['actuator']
        dt = self.native_dt
        actions = []
        for i, cls in enumerate((GenericHeatAction, EcrhAction)):
            location, width = a['deposition'][i]
            actions.append(cls(
                min=[a['lower_MW'][i] * 1e6, location, width],
                max=[a['upper_MW'][i] * 1e6, location, width],
                # Gym-TORAX's rate is per invocation, not per second.
                ramp_rate=[a['ramp_MW_per_s'][i] * dt * 1e6, 0.0, 0.0],
            ))
        return actions

    def _define_observation_space(self):
        # This is a trusted diagnostic extraction. The controller receives only the
        # explicitly defined delayed/noisy scalar observation in eval.py.
        return AllObservation()

    def _compute_reward(self, state, next_state, action):
        # Upstream example's weighted reward is not this task's scoring function.
        return 0.0

    def advance(self, power_MW):
        requested = np.asarray(power_MW, dtype=float)
        count = int(round(self.case['control_dt_s'] / self.native_dt))
        start = self.previous_power.copy()
        self.last_transition = []
        self.last_domains = []
        for index in range(count):
            # Exactly the same 0.5 s endpoint ramp, split at accepted native
            # boundaries; this is action scheduling, not a replacement solver.
            endpoint = start + (index + 1) / count * (requested - start)
            self._advance_native(endpoint)
            self.last_transition.append(diagnostics(self))
            domain = transport_domain(self)
            self.last_domains.append({'time_s': self.last_transition[-1]['time_s'], 'inputs': domain})
            if any(row['outside_faces'] for row in domain.values()):
                raise ArithmeticError('Native accepted state leaves QLKNN training bounds')
        self.previous_power = requested.copy()
        return self.state

    def _advance_native(self, power_MW):
        d = self.case['actuator']['deposition']
        action = {
            'GenericHeat': np.array([power_MW[0] * 1e6, *d[0]], dtype=np.float64),
            'ECRH': np.array([power_MW[1] * 1e6, *d[1]], dtype=np.float64),
        }
        _, _, terminated, _, info = self.step(action)
        if info['action_clipped']:
            raise ValueError('Unexpected native action clipping: ' + str(info['clipped_actions']))
        app = self.torax_app
        error = app.step_fn.check_for_errors(app.current_sim_state, app.current_sim_output)
        if error != SimError.NO_ERROR:
            raise ArithmeticError(f'TORAX SimError {error}')
        solver = app.current_sim_state.solver_numeric_outputs
        if int(solver.solver_error_state) != 0:
            raise ArithmeticError(f'TORAX nonlinear solver status {int(solver.solver_error_state)}')
        actual_time = float(app.current_sim_state.t)
        if abs(actual_time - self.current_time) > 1e-8:
            # Upstream may retain an old state after failure: never score it as a
            # completed transition, regardless of a finite diagnostic array.
            raise ArithmeticError(f'TORAX stopped at {actual_time}, requested {self.current_time}')
        if terminated and actual_time < self.T - 1e-8:
            raise ArithmeticError('Native environment terminated before the requested horizon')


def scalar(state, name):
    x = np.asarray(state['scalars'][name], dtype=float)
    if x.size != 1 or not np.isfinite(x).all():
        raise ArithmeticError(f'Missing/nonfinite native scalar {name}')
    return float(x.ravel()[0])


def native_pedestal_power(env):
    """Integrate the maintained fixed-pedestal adaptive source at a native state."""
    app = env.torax_app
    core = app.current_sim_state.core_profiles
    if not hasattr(env, '_native_pedestal_integral'):
        rp = app.step_fn.runtime_params_provider(float(app.current_sim_state.t))
        geo = app.step_fn.geometry_provider(float(app.current_sim_state.t))
        boundary = app.current_sim_state.pedestal_transition_state.pedestal_model_output.to_internal_boundary_conditions(
            geo, core_profiles=core, pedestal_profile_form=rp.pedestal.pedestal_profile_form)
        zeros = jnp.zeros_like(core.T_e.value)
        si, se, _, mi, me, _ = ibc.apply_adaptive_source(
            source_T_i=zeros, source_T_e=zeros, source_n_e=zeros,
            source_mat_ii=zeros, source_mat_ee=zeros, source_mat_nn=zeros,
            runtime_params=rp, internal_boundary_conditions=boundary)
        # Materialize TORAX's cached geometry property before tracing.
        _ = geo.drho_norm
        # This task fixes geometry and pedestal targets throughout the episode.
        env._native_pedestal_integral = jax.jit(lambda ti, te: math_utils.cell_integration(
            si + mi * ti + se + me * te, geo))
    return float(env._native_pedestal_integral(core.T_i.value, core.T_e.value))


def diagnostics(env: CoreHeatEnv):
    """Read native postprocessing/profile outputs, retaining their native units."""
    s = env.state
    core = env.torax_app.current_sim_state.core_profiles
    post = env.torax_app.current_sim_output
    return {
        'time_s': float(env.torax_app.current_sim_state.t),
        'P_fusion_W': scalar(s, 'P_fusion'),
        'P_ohmic_W': scalar(s, 'P_ohmic_e'),
        'P_auxiliary_W': scalar(s, 'P_aux_total'),
        'P_external_W': scalar(s, 'P_external_total'),
        'P_alpha_W': scalar(s, 'P_alpha_total'),
        'P_pedestal_W': native_pedestal_power(env),
        'P_heat_total_W': scalar(s, 'P_heat_total'),
        'P_SOL_total_W': scalar(s, 'P_SOL_total'),
        'E_external_total_J': scalar(s, 'E_external_total'),
        'W_thermal_total_J': scalar(s, 'W_thermal_total'),
        'T_e_axis_keV': float(core.T_e.value[0]),
        'T_i_axis_keV': float(core.T_i.value[0]),
        'n_e_axis_m3': float(core.n_e.value[0]),
        'q_min': scalar(s, 'q_min'),
        'q95': scalar(s, 'q95'),
        'beta_N': scalar(s, 'beta_N'),
        'temperature_max_keV': float(max(np.max(core.T_e.value), np.max(core.T_i.value))),
        'n_e_line_average_m3': scalar(s, 'n_e_line_avg'),
        'fgw_n_e_line_avg': scalar(s, 'fgw_n_e_line_avg'),
        'solver_error_state': int(env.torax_app.current_sim_state.solver_numeric_outputs.solver_error_state),
    }


@functools.partial(jax.jit, static_argnums=(0,))
def native_model_inputs(native, transport, geo, core_profiles, poloidal_multiplier):
    """Compile the maintained feature preparation; no model formulas are reimplemented."""
    qin = native._prepare_qualikiz_inputs(
        transport=transport, geo=geo, core_profiles=core_profiles,
        poloidal_velocity_multiplier=poloidal_multiplier,
    )
    qin = dataclasses.replace(qin, x=qin.x * qin.epsilon[-1] / qlknn_transport_model._EPSILON_NN)
    model = qlknn_transport_model.get_model(native.path, native.name)
    return model.get_model_inputs_from_qualikiz_inputs(qin)


def transport_domain(env: CoreHeatEnv):
    """Inspect the native QLKNN input preparation and model's published bounds.

    This uses upstream preparation/model objects, with the same aspect-ratio
    coordinate conversion as TORAX 1.4.3's QLKNNTransportModel._combined.
    It never computes replacement transport fluxes.
    """
    app = env.torax_app
    runtime = app.step_fn.runtime_params_provider(app.current_sim_state.t)
    native = app.step_fn.solver.models.transport_model
    model = qlknn_transport_model.get_model(native.path, native.name)
    values = np.asarray(native_model_inputs(
        native, runtime.transport, app.current_sim_state.geometry,
        app.current_sim_state.core_profiles,
        runtime.neoclassical.poloidal_velocity_multiplier), dtype=float)
    rho = np.asarray(app.current_sim_state.geometry.rho_face_norm)
    # Upstream applies exact patches, including their boundary faces, before
    # smoothing; patch-region outputs do not leak into the active region.
    active = (rho > env.native_config['transport']['rho_inner'] + 1e-7) & (rho < env.native_config['transport']['rho_outer'] - 1e-7)
    names = list(model.inputs_and_ranges)
    result = {}
    for i, name in enumerate(names):
        bounds = model.inputs_and_ranges[name]
        if name not in env.core_domain:
            raise ArithmeticError('Missing documented QLKNN11D core bounds: ' + name)
        core_bounds = env.core_domain[name]
        lower, upper = core_bounds['min'], core_bounds['max']
        data = values[active, i]
        if data.size == 0 or not np.isfinite(data).all():
            raise ArithmeticError('Native QLKNN feature preparation is empty/nonfinite: ' + name)
        outside = np.zeros(data.shape, dtype=bool)
        if lower is not None:
            outside |= data < lower
        if upper is not None:
            outside |= data > upper
        result[name] = {
            'min': float(np.min(data)), 'max': float(np.max(data)),
            'trained_min': lower, 'trained_max': upper,
            'combined_model_min': bounds.get('min'), 'combined_model_max': bounds.get('max'),
            'outside_faces': int(np.count_nonzero(outside)),
            'active_faces': int(data.size),
        }
    return result
