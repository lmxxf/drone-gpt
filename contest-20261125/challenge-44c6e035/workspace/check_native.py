#!/usr/bin/env python3
"""Known-answer checks of maintained physics and our geometry conversions."""
import importlib.util
import copy
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation
from simsopt.geo import CurveXYZFourier, CurveLength, LinkingNumber
from simsopt.field import Current, Coil, BiotSavart, coils_via_symmetries

here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('c0_eval', here / 'eval.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def main():
    radius = 0.50
    amps = 1e5
    curve = CurveXYZFourier(512, 1)
    curve.set('xc(1)', radius)
    curve.set('ys(1)', radius)
    current = Current(amps)
    b = BiotSavart([Coil(curve, current)])
    b.set_points(np.array([[0., 0., 0.]]))
    expected = np.array([0., 0., 4 * np.pi * 1e-7 * amps / (2 * radius)])
    actual = b.B()[0]
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-14)
    current.local_x = np.array([0.])
    assert np.array_equal(b.B(), np.zeros((1, 3)))
    current.local_x = np.array([2 * amps])
    assert np.allclose(b.B()[0], 2 * expected, rtol=1e-12, atol=1e-14)
    current.local_x = np.array([amps])
    translation = np.array([0.31, -0.72, 1.11])
    rotvec = np.array([0.11, -0.20, 0.30])
    rotated = e.rigid_transform(curve, translation, rotvec, np.zeros(3), 1, 512)
    br = BiotSavart([Coil(rotated, current)])
    br.set_points(translation.reshape(1, 3))
    expected_r = Rotation.from_rotvec(rotvec).apply(expected)
    assert np.allclose(br.B()[0], expected_r, rtol=1e-12, atol=1e-14)
    assert math.isclose(CurveLength(rotated).J(), 2 * np.pi * radius, rel_tol=1e-12)
    v, a, j = e.curve_derivative_bounds(curve, 1)
    assert v >= max(np.linalg.norm(curve.gammadash(), axis=1))
    assert a >= max(np.linalg.norm(curve.gammadashdash(), axis=1))
    # A simple circle has no eligible pair within 1 m for 4 m arc separation.
    simple = CurveXYZFourier(4096, 2)
    simple.set('xc(1)', 1.)
    simple.set('ys(1)', 1.)
    simple_lower = e.self_distance_lower(simple, 2, 4., 1.)
    assert .99 < simple_lower <= 1.
    # A twice-traversed circle has coincident nonlocal points and must fail.
    doubled = CurveXYZFourier(4096, 2)
    doubled.set('xc(2)', 1.)
    doubled.set('ys(2)', 1.)
    doubled_lower = e.self_distance_lower(doubled, 2, 4., 1.)
    assert -.01 < doubled_lower <= 0.
    # A regular figure eight has nonzero speed but an actual nonlocal crossing.
    crossing = CurveXYZFourier(4096, 2)
    crossing.set('xs(1)', 3.)
    crossing.set('ys(2)', 3.)
    assert min(np.linalg.norm(crossing.gammadash(), axis=1)) > 1.
    crossing_lower = e.self_distance_lower(crossing, 2, 4., 1.)
    assert -.04 < crossing_lower <= 0.
    # Analytically linked perpendicular circles, and a translated unlinked pair.
    xy = CurveXYZFourier(512, 1)
    xy.set('xc(1)', 1.)
    xy.set('ys(1)', 1.)
    xz = CurveXYZFourier(512, 1)
    xz.set('xc(0)', 1.)
    xz.set('xc(1)', 1.)
    xz.set('zs(1)', 1.)
    assert LinkingNumber([xy, xz]).J() == 1
    xz.set('xc(0)', 3.)
    assert LinkingNumber([xy, xz]).J() == 0
    # Exact native symmetry materialization must reproduce field and positions.
    assets = here.parent / 'environment'
    if not assets.exists():
        assets = here / 'assets'
    case = json.loads((assets / 'case.json').read_text())
    data = e.read_submission(assets / 'reference/coils_nominal.json', case)
    base, currents, coils = e.reconstruct(data, 256)
    raw = coils_via_symmetries(base, currents, 2, True)
    native = BiotSavart(raw)
    explicit = []
    for original in raw:
        c = e.materialize(original, data['order'], 256)
        assert np.allclose(c.gamma(), original.curve.gamma(), rtol=1e-12, atol=1e-13)
        explicit.append(Coil(c, original.current))
    points = np.array([[10., 0., 0.], [0., 10., 0.], [9., 2., .1]])
    native.set_points(points)
    clone = BiotSavart(explicit)
    clone.set_points(points)
    assert np.allclose(native.B(), clone.B(), rtol=1e-12, atol=1e-13)
    canonical = BiotSavart(coils)
    canonical.set_points(points)
    assert np.allclose(native.B(), canonical.B(), rtol=1e-12, atol=1e-13)
    # Finite-build circles have an analytic on-axis field: an independent
    # answer for current splitting, section offsets and frame orientation.
    pack_case = copy.deepcopy(case)
    pack_case['winding_pack'].update(width_m=.12, height_m=.20)
    filaments = e.pack_filaments(curve, [0.]*5, pack_case, 3)
    finite = BiotSavart([Coil(f, Current(amps*f.quadrature_weight)) for f in filaments])
    finite.set_points(np.zeros((1, 3)))
    coordinates = np.array([f.gamma()[0] for f in filaments])
    radii = np.hypot(coordinates[:, 0], coordinates[:, 1])
    z = coordinates[:, 2]
    # Check the specified rectangle independently of the native field. Merely
    # using measured radii in the analytic sum would miss a wrong pack width.
    assert np.allclose(np.sort(radii), np.sort(np.repeat(radius + np.array([-np.sqrt(3/5),0.,np.sqrt(3/5)])*.12/2, 3)),
                       rtol=1e-12, atol=1e-14)
    assert np.allclose(np.sort(z), np.sort(np.tile(np.array([-np.sqrt(3/5),0.,np.sqrt(3/5)])*.20/2, 3)),
                       rtol=1e-12, atol=1e-14)
    weights = np.outer([5/18,4/9,5/18], [5/18,4/9,5/18]).ravel()
    assert np.allclose([f.quadrature_weight for f in filaments], weights, rtol=1e-14)
    assert abs(sum(f.quadrature_weight for f in filaments)-1.) < 1e-14
    analytic = 2*np.pi*1e-7*amps*np.sum(weights*radii**2/(radii**2+z**2)**1.5)
    assert np.allclose(finite.B()[0], [0., 0., analytic], rtol=1e-12, atol=1e-14)
    # Reusing one native dependency graph across assembly cases must invalidate
    # the field cache. Translate the circular pack along its axis, with a new
    # analytic answer, then restore it without rebuilding BiotSavart.
    original_coefficients = curve.local_full_x.copy()
    curve.set('zc(0)', .15)
    translated_analytic = 2*np.pi*1e-7*amps*np.sum(weights*radii**2/(radii**2+(z+.15)**2)**1.5)
    assert np.allclose(finite.B()[0], [0.,0.,translated_analytic], rtol=1e-12, atol=1e-14)
    curve.local_full_x = original_coefficients
    assert np.allclose(finite.B()[0], [0.,0.,analytic], rtol=1e-12, atol=1e-14)
    moved_filaments = e.pack_filaments(rotated, [0.]*5, pack_case, 3)
    moved_pack = BiotSavart([Coil(f, Current(amps*f.quadrature_weight)) for f in moved_filaments])
    moved_pack.set_points(translation.reshape(1, 3))
    assert np.allclose(moved_pack.B()[0], Rotation.from_rotvec(rotvec).apply([0.,0.,analytic]),
                       rtol=1e-12, atol=1e-14)
    pack_case['winding_pack'].update(width_m=0., height_m=0.)
    zero = e.pack_filaments(curve, [.3,.1,0.,0.,0.], pack_case, 3)
    finite_zero = BiotSavart([Coil(f, Current(amps*f.quadrature_weight)) for f in zero])
    finite_zero.set_points(np.zeros((1, 3)))
    assert np.allclose(finite_zero.B()[0], expected, rtol=1e-12, atol=1e-14)
    # One sinusoidal twist on a circle: d(alpha)/ds = a*cos(2*pi*t)/R.
    frame = zero[0].framedcurve
    assert np.allclose(abs(np.asarray(frame.frame_torsion())),
                       abs(.1*np.cos(2*np.pi*curve.quadpoints)/radius), atol=1e-12)
    # Nonzero, different rotations must follow their packs through native
    # symmetry expansion and remain invariant under base-array permutation.
    probe = copy.deepcopy(data)
    for i, row in enumerate(probe['coils']):
        row['rotation'] = [.15*i, .05, -.03, .02, 0.]
    base, currents, physical = e.reconstruct(probe, 64)
    base_filaments, split = [], []
    for c, current, row in zip(base, currents, probe['coils']):
        fs = e.pack_filaments(c, row['rotation'], case, 3)
        base_filaments.extend(fs)
        split.extend(Current(current.get_value()*f.quadrature_weight) for f in fs)
    full = BiotSavart(coils_via_symmetries(base_filaments, split, 2, True))
    def pack_field(physical):
        cs = [Coil(f, Current(c.current.get_value()*f.quadrature_weight)) for c in physical
              for f in e.pack_filaments(c.curve, c.pack_rotation, case, 3)]
        b = BiotSavart(cs); b.set_points(points)
        return b.B().copy()
    full.set_points(points)
    reference_pack_field = pack_field(physical)
    assert np.allclose(reference_pack_field, full.B(), rtol=1e-11, atol=1e-12)
    probe['coils'] = [probe['coils'][i] for i in (0,3,1,2)]
    assert np.allclose(pack_field(e.reconstruct(probe, 64)[2]), reference_pack_field,
                       rtol=1e-13, atol=1e-13)
    assert e.score({'x': 1., 'J': 2.}, 2., {'x': {'sense': 'upper', 'threshold': 1., 'scale': 1.}})[0] == .75
    assert e.score({'x': 1., 'J': 1.}, 2., {'x': {'sense': 'upper', 'threshold': 1., 'scale': 1.}})[0] > .75
    assert e.score({'x': 2., 'J': 0.}, 2., {'x': {'sense': 'upper', 'threshold': 1., 'scale': 1.}})[0] < .5
    with tempfile.TemporaryDirectory(dir='/workspace') as tmp:
        malformed = Path(tmp) / 'bad.json'
        malformed.write_text('{"nfp":2,"nfp":2,"order":8,"coils":[]}')
        try:
            e.read_submission(malformed, case)
        except e.InvalidSubmission:
            pass
        else:
            raise AssertionError('duplicate JSON key accepted')
        enormous = copy.deepcopy(data)
        enormous['coils'][0]['x'][0] = 10**400
        malformed.write_text(json.dumps(enormous))
        try:
            e.read_submission(malformed, case)
        except e.InvalidSubmission:
            pass
        else:
            raise AssertionError('400-digit coefficient accepted')
        malformed.write_text('['*2000 + '0' + ']'*2000)
        try:
            e.read_submission(malformed, case)
        except e.InvalidSubmission:
            pass
        else:
            raise AssertionError('excessive JSON nesting accepted')
        for rotation in ([False,0.,0.,0.,0.], [0.]*4, [1e50,0.,0.,0.,0.]):
            hostile = copy.deepcopy(data)
            hostile['coils'][0]['rotation'] = rotation
            malformed.write_text(json.dumps(hostile))
            try:
                e.read_submission(malformed, case)
            except e.InvalidSubmission:
                pass
            else:
                raise AssertionError('invalid material-frame coefficients accepted')
        hostile = copy.deepcopy(data)
        hostile['coils'][0]['width_m'] = 0.
        malformed.write_text(json.dumps(hostile))
        try:
            e.read_submission(malformed, case)
        except e.InvalidSubmission:
            pass
        else:
            raise AssertionError('candidate-controlled pack thickness accepted')
    report = {'status': 'passed', 'circle_B_T': actual.tolist(),
              'analytic_relative_error': float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)),
              'rigid_B_T': br.B()[0].tolist(),
              'self_geometry_lower_m': {'simple_circle': simple_lower,
                                        'double_cover': doubled_lower,
                                        'figure_eight': crossing_lower},
              'checks': ['analytic_circle', 'zero_current', 'current_scaling', 'rigid_covariance',
                         'nonlocal_self_contact', 'native_known_linking',
                         'symmetry_materialization', 'score_contract', 'duplicate_key_rejection',
                         'hostile_numeric_and_nesting_rejection',
                         'finite_pack_analytic_circle', 'zero_build_limit',
                         'analytic_material_twist', 'pack_symmetry_and_permutation',
                         'finite_pack_rigid_covariance', 'hostile_rotation_and_thickness_rejection',
                         'pack_field_cache_tracks_assembly_changes']}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
