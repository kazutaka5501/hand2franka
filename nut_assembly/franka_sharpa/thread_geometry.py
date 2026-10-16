"""Threaded bolt and nut meshes for the screwing stage, generated from the measured dimensions.

No STL of the print is available, so the thread is an ISO-style 60 deg profile with the measured outer diameter
(about 38 mm) and pitch (about 6 mm). Both parts are right-handed: seen from above, turning the nut clockwise
screws it down.

The thread surface is written as a radius over (angle, height), r = f(z - P * theta / 2pi), which a 60 deg profile
allows (no overhang). The nut's internal thread is the bolt's surface pushed out by a radial clearance, as on a
printed part. Only numpy here, so the MuJoCo renderer can use it too.

Frames: bolt origin at the centre of the head's underside (resting on the table), z up along the axis.
Nut origin at its centre, z along the axis, flats facing +-x, as in ``nut_geometry``.
"""

import numpy as np

from franka_sharpa import nut_geometry

PITCH = 0.006
MAJOR_DIAMETER = 0.038
# ISO basic profile: depth 5H/8 with H = 0.866 P, crest flat P/8, root flat P/4
THREAD_DEPTH = 0.625 * 0.8660254 * PITCH
MINOR_DIAMETER = MAJOR_DIAMETER - 2 * THREAD_DEPTH
CLEARANCE = 0.001  # radial, nut thread to bolt thread; 0.5 mm jammed when the nut started 10 deg tilted

BOLT_HEAD_ACROSS_FLATS = 0.051
BOLT_HEAD_HEIGHT = 0.025
BOLT_SHANK_LENGTH = 0.060
BOLT_LENGTH = BOLT_HEAD_HEIGHT + BOLT_SHANK_LENGTH
CHAMFER = 0.003  # 45 deg lead-in on the bolt tip and on both ends of the nut bore

NUM_ANGLES = 192  # multiple of 12: every hexagon corner is a sample angle
DZ = PITCH / 32


def profile(u: np.ndarray) -> np.ndarray:
    """External thread radius at axial phase ``u`` (any real, period ``PITCH``), crest centred at u = 0."""
    w = np.abs((u + 0.5 * PITCH) % PITCH - 0.5 * PITCH)  # distance from the crest centre, 0..P/2
    crest_half, flank = PITCH / 16, 5 * PITCH / 16
    r_major = 0.5 * MAJOR_DIAMETER
    return np.clip(r_major - (w - crest_half) * np.tan(np.pi / 3), r_major - THREAD_DEPTH, r_major)


def _angles():
    return np.arange(NUM_ANGLES) * 2 * np.pi / NUM_ANGLES


def _grid_faces(num_rings: int, offset: int, outward: bool) -> list:
    """Quads between consecutive rings of ``NUM_ANGLES`` vertices, as triangles."""
    faces = []
    n = NUM_ANGLES
    for i in range(num_rings - 1):
        a, b = offset + i * n, offset + (i + 1) * n
        for k in range(n):
            j = (k + 1) % n
            if outward:
                faces += [[a + k, a + j, b + j], [a + k, b + j, b + k]]
            else:
                faces += [[a + k, b + j, a + j], [a + k, b + k, b + j]]
    return faces


def bolt_shank_mesh():
    """Closed threaded cylinder from inside the head (z = head height / 2) to the tip, ``(vertices, faces)``."""
    theta = _angles()
    z_bottom, z_top = 0.5 * BOLT_HEAD_HEIGHT, BOLT_LENGTH
    zs = np.arange(z_bottom, z_top + 0.5 * DZ, DZ)
    zs[-1] = z_top
    zz, tt = np.meshgrid(zs, theta, indexing="ij")
    r = profile(zz - PITCH * tt / (2 * np.pi))
    r = np.minimum(r, 0.5 * MINOR_DIAMETER + (z_top - zz))  # tip chamfer
    vertices = np.stack([r * np.cos(tt), r * np.sin(tt), zz], axis=-1).reshape(-1, 3)
    faces = _grid_faces(len(zs), 0, outward=True)
    bottom, top = len(vertices), len(vertices) + 1
    vertices = np.vstack([vertices, [0, 0, z_bottom], [0, 0, z_top]])
    last = (len(zs) - 1) * NUM_ANGLES
    for k in range(NUM_ANGLES):
        j = (k + 1) % NUM_ANGLES
        faces += [[bottom, j, k], [top, last + k, last + j]]
    return vertices, np.array(faces, dtype=np.int64)


def bolt_head_vertices():
    """Hexagonal head as a convex prism, ``(12, 3)``."""
    phi = np.arange(6) * np.pi / 3 + np.pi / 6  # corners; flats face +-x
    r = 0.5 * BOLT_HEAD_ACROSS_FLATS / np.cos(np.pi / 6)
    ring = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)
    return np.vstack([np.hstack([ring, np.zeros((6, 1))]), np.hstack([ring, np.full((6, 1), BOLT_HEAD_HEIGHT)])])


def bolt_head_faces():
    faces = [[0, 5, 4, 3, 2, 1], [6, 7, 8, 9, 10, 11]]
    faces += [[k, (k + 1) % 6, 6 + (k + 1) % 6, 6 + k] for k in range(6)]
    return faces


def nut_mesh():
    """Closed hex nut with an internal thread, ``(vertices, faces)``.

    At height z (nut frame) and angle theta the bore radius is ``profile(z - P theta / 2pi) + CLEARANCE``, so the
    nut mates with the bolt when its frame sits at a height that is a multiple of the pitch on the bolt axis with
    zero yaw (or any (yaw, height) pair on that helix).
    """
    theta = _angles()
    h = 0.5 * nut_geometry.THICKNESS
    zs = np.arange(-h, h + 0.5 * DZ, DZ)
    zs[-1] = h
    zz, tt = np.meshgrid(zs, theta, indexing="ij")
    r_in = profile(zz - PITCH * tt / (2 * np.pi)) + CLEARANCE
    end_dist = h - np.abs(zz)
    r_in = np.maximum(r_in, 0.5 * MAJOR_DIAMETER + CLEARANCE + 0.5e-3 - end_dist)  # countersinks
    inner = np.stack([r_in * np.cos(tt), r_in * np.sin(tt), zz], axis=-1).reshape(-1, 3)
    r_out = nut_geometry._outer_radius(theta)
    outer_bottom = np.stack([r_out * np.cos(theta), r_out * np.sin(theta), np.full(NUM_ANGLES, -h)], axis=1)
    outer_top = np.stack([r_out * np.cos(theta), r_out * np.sin(theta), np.full(NUM_ANGLES, h)], axis=1)
    vertices = np.vstack([inner, outer_bottom, outer_top])

    n, rings = NUM_ANGLES, len(zs)
    faces = _grid_faces(rings, 0, outward=False)  # bore wall, normal towards the axis
    ob, ot = rings * n, rings * n + n
    it = (rings - 1) * n
    for k in range(n):
        j = (k + 1) % n
        faces += [[ob + k, ob + j, ot + j], [ob + k, ot + j, ot + k]]  # outer wall
        faces += [[it + k, ot + k, ot + j], [it + k, ot + j, it + j]]  # top annulus
        faces += [[k, j, ob + j], [k, ob + j, ob + k]]  # bottom annulus
    return vertices, np.array(faces, dtype=np.int64)
