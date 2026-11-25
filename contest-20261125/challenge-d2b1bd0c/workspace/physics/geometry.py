"""Fixed-volume boundary construction; no equilibrium or stability physics.

Six symmetric shape coefficients; minor scale is solved at fixed toroidal
volume using Shapely polygon area/centroid and SciPy's bracketed root solver.
Submission bounds are defined by the evaluation protocol.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from shapely.geometry import Polygon


@dataclass(frozen=True)
class Shape:
    kappa: float = 1.8
    delta: float = 0.4
    c2: float = 0.0
    c3: float = 0.0
    s2: float = 0.0
    s3: float = 0.0


def boundary(shape, volume=240.0, npoints=2048):
    values = np.array(list(vars(shape).values()), dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(volume) or volume <= 0:
        raise ValueError("Shape and volume must be finite, with positive volume")
    if not 1.0 <= shape.kappa <= 2.5 or abs(shape.delta) >= 0.8:
        raise ValueError("Elongation or triangularity outside diagnostic bounds")
    if np.max(np.abs(values[2:])) > 0.2:
        raise ValueError("Shape harmonic outside diagnostic bounds")
    theta = np.linspace(0, 2*np.pi, npoints, endpoint=False)
    phase = theta + np.arcsin(shape.delta)*np.sin(theta)
    phase += shape.c2*np.sin(2*theta) + shape.c3*np.sin(3*theta)
    radial = np.cos(phase)
    vertical = shape.kappa*(np.sin(theta) + shape.s2*np.sin(2*theta)
                            + shape.s3*np.sin(3*theta))

    def polygon(scale):
        points = np.column_stack((4.0+scale*radial, scale*vertical))
        domain = Polygon(points)
        if not domain.is_valid or not domain.exterior.is_simple or domain.area <= 0:
            raise ValueError("Shape does not form a simple closed plasma section")
        if points[:, 0].min() <= 0:
            raise ValueError("Shape intersects the symmetry axis")
        return points, domain

    def toroidal_volume(scale):
        _, domain = polygon(scale)
        return 2*np.pi*domain.centroid.x*domain.area

    scale = brentq(lambda a: toroidal_volume(a)-volume, 0.5, 2.2, xtol=1e-13)
    points, domain = polygon(scale)
    return points, {"minor_scale_m": scale, "volume_m3": toroidal_volume(scale),
                    "area_m2": domain.area, "perimeter_m": domain.length,
                    "bounds_m": list(domain.bounds)}


def verify_circle():
    # Analytic torus: V=2*pi^2*R*a^2. The regular-polygon correction is exact.
    npoints = 2048
    target = 2*np.pi**2*4.0*1.3**2
    _, info = boundary(Shape(kappa=1, delta=0), volume=target, npoints=npoints)
    polygon_correction = np.sqrt((2*np.pi/npoints)/np.sin(2*np.pi/npoints))
    expected = 1.3*polygon_correction
    if abs(info["minor_scale_m"]-expected) > 1e-11:
        raise AssertionError((info["minor_scale_m"], expected))
    return {"analytic_minor_scale_m": expected, "computed": info}


if __name__ == "__main__":
    import json
    print(json.dumps(verify_circle(), indent=2))
