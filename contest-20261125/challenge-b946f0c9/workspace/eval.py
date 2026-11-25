#!/usr/bin/env python3
"""N0: input/CSG/native OpenMC/output/statistics wrapper, not a transport solver."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np
from scipy.stats import t as student_t


class InvalidSubmission(ValueError):
    pass


class Unresolved(RuntimeError):
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics


class EvaluatorError(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise InvalidSubmission(f'duplicate JSON key: {key}')
        out[key] = value
    return out


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=duplicate_keys,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          InvalidSubmission(f'nonfinite JSON constant: {value}')))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def submission_snapshot(path):
    """The sole snapshot owner rejects links, devices and FIFOs before reading."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise InvalidSubmission(f'cannot open regular submission: {error}') from error
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
            raise InvalidSubmission('submission must be a regular file of at most 1 MiB')
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise InvalidSubmission('submission is larger than 1 MiB')
    return raw


def region_ids():
    return [f'L{layer}_Y{iy}_Z{iz}' for layer in range(6)
            for iy in range(4) for iz in range(4)]


def parse_design(path, case):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise InvalidSubmission('submission must be a file of at most 1 MiB')
    try:
        raw = read_json(path)
    except (OSError, json.JSONDecodeError, UnicodeError, RecursionError) as error:
        raise InvalidSubmission(f'cannot read design: {error}') from error
    if not isinstance(raw, dict) or set(raw) != {'schema_version', 'case_id', 'regions'}:
        raise InvalidSubmission('top-level keys do not match the design contract')
    if raw['schema_version'] != 'fusionx-n0-design/v1' or raw['case_id'] != case['case_id']:
        raise InvalidSubmission('wrong schema_version or case_id')
    if not isinstance(raw['regions'], list) or len(raw['regions']) != 96:
        raise InvalidSubmission('exactly 96 region assignments required')
    found = {}
    for item in raw['regions']:
        if not isinstance(item, dict) or set(item) != {'id', 'breeder_fraction', 'li6_atom_fraction'}:
            raise InvalidSubmission('region keys do not match the design contract')
        rid = item['id']
        if not isinstance(rid, str) or rid not in region_ids() or rid in found:
            raise InvalidSubmission('unknown or repeated region ID')
        for key, low, high in [('breeder_fraction', .20, .60), ('li6_atom_fraction', .05, .90)]:
            value = item[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InvalidSubmission(f'{rid}/{key}: finite numeric value required')
            if not math.isfinite(value) or not low <= value <= high:
                raise InvalidSubmission(f'{rid}/{key}: out of permitted range')
        found[rid] = item
    return [found[rid] for rid in region_ids()]


def overlap(a0, a1, b0, b1):
    return max(0., min(a1, b1) - max(a0, b0))


def region_volume(iy, iz):
    """Exact rectangular inventory geometry, including fixed root-level cutouts."""
    y0, y1 = -40 + 20 * iy, -20 + 20 * iy
    z0, z1 = -40 + 20 * iz, -20 + 20 * iz
    area = 400. - overlap(y0, y1, 7.5, 18.5) * overlap(z0, z1, -5.5, 5.5)
    area -= 20 * (overlap(y0, y1, -21, -19) + overlap(y0, y1, 19, 21))
    return 10 * area


def uniform(case):
    return [{'id': rid, 'breeder_fraction': .40, 'li6_atom_fraction': .30}
            for rid in region_ids()]


def add_natural_element(openmc, material, symbol, amount=1.):
    # Explicit maintained abundances prevent add_element silently removing an isotope
    # that is absent from the selected library and renormalizing the remaining ones.
    items = [(name, fraction) for name, fraction in openmc.data.NATURAL_ABUNDANCE.items()
             if re.fullmatch(symbol + r'[0-9]+', name)]
    if not items or abs(sum(fraction for _, fraction in items) - 1.) > 1e-6:
        raise EvaluatorError(f'native natural abundance unavailable: {symbol}')
    for name, fraction in sorted(items):
        material.add_nuclide(name, amount * fraction)


def lithium_bed(openmc, enrichment):
    material = openmc.Material(name=f'Li4SiO4-bed-{enrichment:.16g}')
    material.add_nuclide('Li6', 4 * enrichment)
    material.add_nuclide('Li7', 4 * (1 - enrichment))
    add_natural_element(openmc, material, 'Si', 1.)
    add_natural_element(openmc, material, 'O', 4.)
    reference = openmc.Material(name='Li4SiO4-molar-mass-reference')
    reference.add_nuclide('Li6', 1.2)
    reference.add_nuclide('Li7', 2.8)
    add_natural_element(openmc, reference, 'Si', 1.)
    add_natural_element(openmc, reference, 'O', 4.)
    # Hold formula-unit number density and packing fixed as isotope mass changes.
    # The reference solid density is 2.30 g/cm3 at e=0.30, packing fraction 0.64.
    material.set_density('g/cm3', 1.472 * material.average_molar_mass / reference.average_molar_mass)
    material.temperature = 294.
    return material


def prepare_model(design, case, source_kind, seed, data_xml, *, particles=None, batches=None):
    import openmc
    if openmc.__version__ != case['openmc']['version']:
        raise EvaluatorError(f'OpenMC {openmc.__version__} differs from pinned {case["openmc"]["version"]}')
    openmc.reset_auto_ids()
    fe, he, be, tungsten = [openmc.Material(name=name)
                            for name in ['natural-Fe', 'He4', 'Be9-bed', 'natural-W']]
    for mat, density, name, element in [(fe, 7.87, 'Fe', True), (he, .006, 'He4', False),
                                       (be, 1.184, 'Be9', False), (tungsten, 19.3, 'W', True)]:
        if element:
            add_natural_element(openmc, mat, name)
        else:
            mat.add_nuclide(name, 1.)
        mat.set_density('g/cm3', density)
        mat.temperature = 294.
    shield = openmc.Material.mix_materials([fe, he], [.8, .2], 'vo', name='fixed-rear-shield')
    shield.temperature = 294.
    xplanes = {x: openmc.XPlane(x0=x, boundary_type='vacuum' if x in (0, 96) else 'transmission')
               for x in [0, .2, 1, 11, 21, 31, 41, 51, 61, 91, 96]}
    yp = {y: openmc.YPlane(y0=y, boundary_type='vacuum' if abs(y) == 40 else 'transmission')
          for y in [-40, -21, -20.5, -20, -19.5, -19, 0, 7.5, 8, 18, 18.5, 19, 19.5, 20, 20.5, 21, 40]}
    zp = {z: openmc.ZPlane(z0=z, boundary_type='vacuum' if abs(z) == 40 else 'transmission')
          for z in [-40, -20, -5.5, -5, 0, 5, 5.5, 20, 40]}

    def slab(x0, x1):
        return +xplanes[x0] & -xplanes[x1]

    def rect(y0, y1, z0=-40, z1=40):
        return +yp[y0] & -yp[y1] & +zp[z0] & -zp[z1]

    cross = rect(-40, 40)
    port_outer = slab(0, 91) & rect(7.5, 18.5, -5.5, 5.5)
    port_inner = slab(0, 91) & rect(8, 18, -5, 5)
    cells = [openmc.Cell(cell_id=10, name='diagnostic-vacuum', region=port_inner),
             openmc.Cell(cell_id=11, name='port-iron-liner', region=port_outer & ~port_inner, fill=fe),
             openmc.Cell(cell_id=12, name='first-wall', region=slab(0, .2) & cross & ~port_outer, fill=tungsten),
             openmc.Cell(cell_id=13, name='front-structure', region=slab(.2, 1) & cross & ~port_outer, fill=fe),
             openmc.Cell(cell_id=14, name='rear-shield', region=slab(61, 91) & cross & ~port_outer, fill=shield)]
    cool_outer = slab(1, 61) & (rect(-21, -19) | rect(19, 21))
    cool_inner = slab(1, 61) & (rect(-20.5, -19.5) | rect(19.5, 20.5))
    cells += [openmc.Cell(cell_id=15, name='helium-channels', region=cool_inner, fill=he),
              openmc.Cell(cell_id=16, name='coolant-sidewalls', region=cool_outer & ~cool_inner, fill=fe)]
    breeder_cells, inventory = [], {'breeder_volume_cm3': 0., 'be_volume_cm3': 0., 'li6_atoms': 0.}
    mats = [fe, he, be, tungsten, shield]
    for index, item in enumerate(design):
        layer, tile = divmod(index, 16)
        iy, iz = divmod(tile, 4)
        volume = region_volume(iy, iz)
        bed = lithium_bed(openmc, item['li6_atom_fraction'])
        b = item['breeder_fraction']
        mix = openmc.Material.mix_materials([bed, be, fe, he], [b, .75 - b, .15, .10], 'vo', name=item['id'])
        mix.temperature = 294.
        mats.append(mix)
        inventory['breeder_volume_cm3'] += volume * b
        inventory['be_volume_cm3'] += volume * (.75 - b)
        inventory['li6_atoms'] += volume * mix.get_nuclide_atom_densities()['Li6'] * 1e24
        native_cell = openmc.Cell(cell_id=1000 + index, name=item['id'], fill=mix,
                                 region=slab(1 + 10 * layer, 11 + 10 * layer)
                                 & rect(-40 + 20 * iy, -20 + 20 * iy, -40 + 20 * iz, -20 + 20 * iz))
        native_cell.volume = volume
        breeder_cells.append(native_cell)
    breeder_universe = openmc.Universe(universe_id=200, cells=breeder_cells)
    cells.append(openmc.Cell(cell_id=17, name='breeder-container', fill=breeder_universe,
                             region=slab(1, 61) & cross & ~port_outer & ~cool_outer))
    detectors = []
    for iy in range(4):
        for iz in range(4):
            detector = openmc.Cell(cell_id=3000 + 4 * iy + iz, name=f'rear-Y{iy}-Z{iz}',
                                   region=slab(91, 96) & rect(-40 + 20 * iy, -20 + 20 * iy,
                                                           -40 + 20 * iz, -20 + 20 * iz))
            detector.volume = 2000.
            detectors.append(detector)
    cells.extend(detectors)
    material_collection = openmc.Materials(mats)
    material_collection.cross_sections = str(Path(data_xml).resolve())
    settings = openmc.Settings()
    settings.run_mode = 'fixed source'
    settings.batches = batches if batches is not None else case['sampling']['batches_per_seed']
    settings.particles = particles if particles is not None else case['sampling']['particles_per_batch']
    settings.inactive = 0
    settings.seed = seed
    settings.statepoint = {'batches': list(range(1, settings.batches + 1))}
    settings.max_lost_particles = 1
    # Native API requires a positive threshold. Any lost particle still fails
    # this wrapper; 1e-12 is below one history at the fixed allowed run budget.
    settings.rel_max_lost_particles = 1e-12
    settings.photon_transport = False
    settings.temperature = {'default': 294., 'method': 'nearest', 'tolerance': .5}
    angle = (openmc.stats.Monodirectional(reference_uvw=(1., 0., 0.)) if source_kind == 'normal'
             else openmc.stats.PolarAzimuthal(mu=openmc.stats.PowerLaw(0, 1, 1),
                                             phi=openmc.stats.Uniform(0, 2 * math.pi), reference_uvw=(1., 0., 0.),
                                             reference_vwu=(0., 1., 0.)))
    settings.source = openmc.IndependentSource(
        space=openmc.stats.CartesianIndependent(openmc.stats.Discrete([1e-6], [1]),
                                               openmc.stats.Uniform(-40, 40), openmc.stats.Uniform(-40, 40)),
        angle=angle, energy=openmc.stats.Discrete([14.1e6], [1]), strength=1., particle='neutron')
    production = openmc.Tally(tally_id=1, name='lithium-tritium-production')
    production.filters = [openmc.UniverseFilter([breeder_universe])]
    production.scores = ['H3-production']
    production.nuclides = ['Li6', 'Li7']
    production.estimator = 'tracklength'
    fast = openmc.Tally(tally_id=2, name='rear-fast-neutron-flux')
    fast.filters = [openmc.CellFilter(detectors), openmc.EnergyFilter([1e5, 1e9])]
    fast.scores = ['flux']
    fast.estimator = 'tracklength'
    model = openmc.Model(geometry=openmc.Geometry(cells), materials=material_collection,
                         settings=settings, tallies=openmc.Tallies([production, fast]))
    return model, inventory


def require_data_coverage(model, data_xml):
    """Native library metadata/coverage checks; no cross-section generation."""
    import openmc
    library = openmc.data.DataLibrary.from_xml(data_xml)
    for nuclide in {n for material in model.materials for n in material.get_nuclides()}:
        record = library.get_by_material(nuclide, 'neutron')
        if not record:
            raise EvaluatorError(f'neutron cross section missing: {nuclide}')
        path = Path(record['path'])
        if not path.is_file():
            raise EvaluatorError(f'cross-section path is absent: {path}')
        data = openmc.data.IncidentNeutron.from_hdf5(path)
        if not any(abs(float(kt) / openmc.data.K_BOLTZMANN - 294.) <= .5 for kt in data.kTs):
            raise EvaluatorError(f'294 K coverage missing: {nuclide}')


def batch_observations(work, batches):
    import openmc
    previous = np.zeros(17)
    values, artifacts = [], []
    indexed = {}
    for candidate in work.glob('statepoint.*.h5'):
        with openmc.StatePoint(candidate) as state:
            batch = state.current_batch
        if batch in indexed or not 1 <= batch <= batches:
            raise EvaluatorError('duplicate/out-of-range native statepoint batch')
        indexed[batch] = candidate
    for batch in range(1, batches + 1):
        if batch not in indexed:
            raise EvaluatorError(f'current statepoint batch missing: {batch}')
        path = indexed[batch]
        with openmc.StatePoint(path) as state:
            if state.run_mode != 'fixed source' or state.n_realizations != batch:
                raise EvaluatorError(f'invalid statepoint batch normalization: {path}')
            production = state.get_tally(id=1)
            fast = state.get_tally(id=2)
            # Sum NUCLIDE contributions at each batch before estimating variance.
            # StatePoint sum is the cumulative sum of batch-normalized tally values.
            cumulative = np.concatenate(([float(production.sum.sum())], fast.sum.reshape(-1) / 2000.))
            if cumulative.shape != (17,) or not np.isfinite(cumulative).all():
                raise EvaluatorError('unexpected native tally shape/nonfinite values')
            observation = cumulative - previous
            if (observation < -1e-12).any():
                raise EvaluatorError('native cumulative tally decreased beyond roundoff')
            values.append(observation)
            previous = cumulative
        artifacts.append({'path': str(path), 'sha256': sha256(path)})
    return np.asarray(values), artifacts


def summarize_replicates(replicates, family_size=68, *, reference=None):
    data = np.concatenate(replicates)
    if len(replicates) != 2 or any(rep.shape != replicates[0].shape for rep in replicates):
        raise EvaluatorError('two equal-size independent replicates required')
    if data.ndim != 2 or data.shape[1] != 17 or not np.isfinite(data).all():
        raise EvaluatorError('invalid batch observations')
    n = len(data)
    if n < 4:
        raise EvaluatorError('insufficient batch count')
    quantile = float(student_t.ppf(1 - .05 / family_size, n - 1))
    mean, se = data.mean(0), data.std(0, ddof=1) / math.sqrt(n)
    half = quantile * se
    means = [rep.mean(0) for rep in replicates]
    ses = [rep.std(0, ddof=1) / math.sqrt(len(rep)) for rep in replicates]
    disagreement = np.abs(means[0] - means[1])
    allowance = quantile * np.hypot(ses[0], ses[1])
    positive_fraction = (data > 0).mean(0)
    result = {'mean': mean.tolist(), 'standard_error': se.tolist(), 'lower': (mean - half).tolist(),
              'upper': (mean + half).tolist(), 'halfwidth': half.tolist(), 'n_batches': n,
              'quantile': quantile, 'seed_difference': disagreement.tolist(),
              'seed_difference_limit': allowance.tolist(), 'positive_batch_fraction': positive_fraction.tolist()}
    if reference is not None:
        tolerance = np.concatenate(([.005 * reference['mean'][0]], .05 * np.asarray(reference['screening_limits'])))
        result['halfwidth_limit'] = tolerance.tolist()
        if (half > tolerance).any() or (positive_fraction < .90).any() or (disagreement > allowance).any():
            raise Unresolved('batch precision, positive event coverage or independent-seed agreement failed', result)
    return result


def score_result(inventory, limits, outcomes, baseline):
    violations = {key: max(0., value / limits[key] - 1. - 1e-10)
                  for key, value in inventory.items()}
    for source_kind, result in outcomes.items():
        for tile, (value, bound) in enumerate(zip(result['upper'][1:], baseline['cases'][source_kind]['screening_limits'])):
            violations[f'{source_kind}/rear{tile}'] = max(0., value / bound - 1.)
    objective = min(result['lower'][0] for result in outcomes.values())
    reference_objective = baseline['objective_lower']
    if not reference_objective > 0 or not math.isfinite(reference_objective):
        raise EvaluatorError('positive finite reference objective required')
    violations['negative_yield_lower'] = max(0., -objective / reference_objective)
    feasible = not any(value > 0 for value in violations.values())
    if feasible:
        score = .5 + .5 * objective / (objective + reference_objective)
    else:
        score = min(math.nextafter(.5, 0.), .5 * math.exp(-sum(v * v for v in violations.values())))
    return float(score), feasible, float(objective), violations


def load_trusted(root, case_path):
    runtime_path, baseline_path = root / 'runtime-lock.json', root / 'reference' / 'baseline.json'
    try:
        case = read_json(case_path)
        runtime, baseline = read_json(runtime_path), read_json(baseline_path)
    except (InvalidSubmission, OSError, ValueError) as error:
        raise EvaluatorError(f'trusted configuration cannot be read: {error}') from error
    if runtime.get('state') != 'verified' or baseline.get('state') != 'calibrated':
        raise EvaluatorError('N0 runtime or continuous-energy reference not frozen')
    if runtime.get('isolation') != 'clean_trusted_verifier' or '@sha256:' not in runtime.get('image_digest', ''):
        raise EvaluatorError('verified immutable image and clean verifier required')
    if baseline.get('case_sha256') != sha256(case_path) or baseline.get('runtime_lock_sha256') != sha256(runtime_path):
        raise EvaluatorError('reference is for a different case/runtime')
    if baseline.get('reference_sha256') != sha256(root / 'reference_configurations' / 'uniform.json'):
        raise EvaluatorError('reference design mismatch')
    files = runtime.get('verified_files', [])
    required = {runtime['openmc_executable'], str(Path(sys.executable).resolve()), runtime['cross_sections_xml']}
    if not required <= {record['path'] for record in files}:
        raise EvaluatorError('program/interpreter/nuclear data file inventory incomplete')
    for record in files:
        if not Path(record['path']).is_file() or sha256(record['path']) != record['sha256']:
            raise EvaluatorError(f'runtime file mismatch: {record["path"]}')
    xml = ET.parse(runtime['cross_sections_xml'])
    data_paths = {str((Path(runtime['cross_sections_xml']).parent / item.get('path')).resolve())
                  for item in xml.findall('library') if item.get('type') == 'neutron'}
    if not data_paths <= {record['path'] for record in files}:
        raise EvaluatorError('nuclear library file hashes incomplete')
    required_evidence = {'void_source_normalization', 'no_lithium_zero', 'thin_target',
                         'geometry_coverage', 'independent_seed_coverage', 'particle_count_refinement',
                         'reference_replay', 'runtime_isolation'}
    evidence = baseline.get('evidence', {})
    if not required_evidence <= set(evidence):
        raise EvaluatorError('scientific calibration witnesses missing')
    for name in required_evidence:
        record = evidence[name]
        if record.get('status') != 'passed' or not record.get('artifacts'):
            raise EvaluatorError(f'calibration witness not verified: {name}')
        for artifact in record['artifacts']:
            path = root / artifact['path']
            if not path.is_file() or sha256(path) != artifact['sha256']:
                raise EvaluatorError(f'calibration artifact mismatch: {name}')
    return case, runtime, baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('submission', type=Path)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--reward-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--physics-only', action='store_true', help='unranked author measurement, no calibration bypass in test.sh')
    parser.add_argument('--cross-sections', type=Path)
    parser.add_argument('--openmc-executable', type=Path)
    parser.add_argument('--particles', type=int, help='unranked measurement only')
    parser.add_argument('--batches', type=int, help='unranked measurement only')
    parser.add_argument('--source-kind', choices=['normal', 'cosine'], help='one source for an unranked measurement only')
    parser.add_argument('--seed-pair', type=Path, help='trusted official independent pair; absent uses public trial seeds')
    args = parser.parse_args()
    args.reward_dir.mkdir(parents=True, exist_ok=True)
    details = {'status': 'evaluator_error', 'version': 'fusionx-n0-evaluation/v1', 'stages': {}}
    reward = {'reward': 0., 'valid': 0, 'feasible': 0, 'evaluator_error': 0}
    started = time.monotonic()
    try:
        here = Path(__file__).resolve().parent
        # The standard export mounts /tests separately from /app resources.
        # The build owns this copy; official grading never reads candidate assets.
        root = here / 'n0_assets'
        if not root.is_dir():
            root = here.parent / 'environment'
        if not root.is_dir():
            root = here
        case_path = root / 'case.json'
        if args.physics_only:
            case, runtime, baseline = read_json(case_path), None, None
            if not args.cross_sections or not args.openmc_executable:
                raise EvaluatorError('author measurement requires explicit native executable and nuclear XML')
            data_xml, executable = args.cross_sections.resolve(), args.openmc_executable.resolve()
        else:
            if args.cross_sections or args.openmc_executable or args.particles or args.batches or args.source_kind:
                raise EvaluatorError('official evaluator accepts no native configuration overrides')
            case, runtime, baseline = load_trusted(root, case_path)
            data_xml, executable = Path(runtime['cross_sections_xml']), Path(runtime['openmc_executable'])
        if args.threads < 1 or args.threads > 8:
            raise EvaluatorError('thread count must be 1..8')
        if not data_xml.is_file() or not executable.is_file():
            raise EvaluatorError('native OpenMC executable or nuclear XML missing')
        if args.workdir.exists() and any(args.workdir.iterdir()):
            raise EvaluatorError('workdir must be new or empty')
        args.workdir.mkdir(parents=True, exist_ok=True)
        snapshot = args.workdir / 'submission.json'
        raw = submission_snapshot(args.submission)
        snapshot.write_bytes(raw)
        design = parse_design(snapshot, case)
        details['submission_sha256'] = sha256(snapshot)
        details['case_sha256'] = sha256(case_path)
        outcomes = {}
        deadline = started + case['evaluation_timeout_seconds']
        seeds = read_json(args.seed_pair) if args.seed_pair else case['sampling']['seeds']
        if (not isinstance(seeds, list) or len(seeds) != 2 or seeds[0] == seeds[1]
                or any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 < seed < 2**63 for seed in seeds)):
            raise EvaluatorError('two distinct positive native RNG seeds below 2**63 required')
        for source_kind in [args.source_kind] if args.source_kind else case['sources']:
            histories = []
            stage = {'runs': []}
            details['stages'][source_kind] = stage
            for seed in seeds:
                model, inventory = prepare_model(design, case, source_kind, seed, data_xml,
                                                  particles=args.particles, batches=args.batches)
                require_data_coverage(model, data_xml)
                run = args.workdir / f'{source_kind}-seed{seed}'
                run.mkdir()
                model.export_to_xml(run)
                native_env = {key: value for key, value in os.environ.items()
                              if key not in {'PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD', 'OPENMC_CROSS_SECTIONS'}}
                native_env['OPENMC_CROSS_SECTIONS'] = str(data_xml)
                native_env['OMP_NUM_THREADS'] = str(args.threads)
                with (run / 'openmc.log').open('w') as log:
                    process = subprocess.run([str(executable), '-s', str(args.threads)], cwd=run,
                                             env=native_env, stdout=log, stderr=subprocess.STDOUT,
                                             timeout=max(.1, deadline - time.monotonic()))
                if process.returncode:
                    raise EvaluatorError(f'native transport failed; inspect {run}/openmc.log')
                log_text = (run / 'openmc.log').read_text().lower()
                if 'lost particle' in log_text or 'could not be located' in log_text or list(run.glob('particle*.h5')):
                    raise EvaluatorError('particle location failure in fixed evaluator geometry')
                observations, artifacts = batch_observations(run, model.settings.batches)
                histories.append(observations)
                stage['runs'].append({'seed': seed, 'particles_per_batch': model.settings.particles,
                                      'batches': model.settings.batches, 'statepoints': artifacts,
                                      'inputs': [{'path': str(run / name), 'sha256': sha256(run / name)}
                                                 for name in ['materials.xml', 'geometry.xml', 'settings.xml', 'tallies.xml']]})
                write_json(args.reward_dir / 'details.json', details)
            outcomes[source_kind] = summarize_replicates(histories,
                                                          reference=None if baseline is None else baseline['cases'][source_kind])
            stage['summary'] = outcomes[source_kind]
        details['inventory'] = inventory
        details['outcomes'] = outcomes
        if args.physics_only:
            details['status'] = 'not_ranked'
            details['reason'] = 'actual native measurement; no calibrated baseline or official score'
        else:
            score, feasible, objective, violations = score_result(inventory, baseline['inventory_limits'], outcomes, baseline)
            details.update(status='feasible' if feasible else 'infeasible', objective_lower=objective, violations=violations)
            details['screening'] = {
                source: {'limits': baseline['cases'][source]['screening_limits'],
                         'upper_margin': (np.asarray(baseline['cases'][source]['screening_limits'])
                                          - np.asarray(result['upper'][1:])).tolist()}
                for source, result in outcomes.items()}
            reward.update(reward=score, valid=1, feasible=int(feasible))
    except InvalidSubmission as error:
        details.update(status='invalid_submission', reason=str(error))
    except (Unresolved, subprocess.TimeoutExpired) as error:
        details.update(status='candidate_unresolved', reason=str(error))
        if isinstance(error, Unresolved) and error.diagnostics is not None:
            details['unresolved_diagnostics'] = error.diagnostics
    except Exception as error:
        details.update(status='evaluator_error', reason=str(error), traceback=traceback.format_exc())
        reward['evaluator_error'] = 1
    details['elapsed_seconds'] = time.monotonic() - started
    write_json(args.reward_dir / 'details.json', details)
    write_json(args.reward_dir / 'reward.json', reward)


if __name__ == '__main__':
    main()
