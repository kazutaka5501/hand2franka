"""Geometry of the 3D-printed hex nut, measured with a tape measure (2026-10-15).

Readings from the photos, which carry a few millimetres of parallax error:
    * width over the top face, corner to corner: about 59-60 mm
    * thickness along the axis (nut lying on its side): about 33 mm
    * matching bolt: thread outer diameter about 38 mm, pitch about 6 mm, overall height with the nut about 90 mm
The across-flats width follows from the corner-to-corner one (x cos 30 deg). The hole is modelled as a smooth
bore between the bolt's thread root and crest diameters; the thread itself does not matter for picking the nut up.

Only numpy here, so the MuJoCo renderer can use it from outside the Isaac Lab environment.
Nut frame: origin at the centre, z along the axis, flats facing +-x.
"""

import numpy as np

ACROSS_FLATS = 0.051
ACROSS_CORNERS = ACROSS_FLATS / np.cos(np.pi / 6)
THICKNESS = 0.033
BORE_DIAMETER = 0.036
# PLA print with walls and partial infill, about 40 cm^3 of material at roughly 0.9 g/cm^3 effective density
MASS = 0.035

# centre height above the table in the two ways the nut rests: on a hex face (axis vertical) or on a side flat
REST_HEIGHT_FLAT = 0.5 * THICKNESS
REST_HEIGHT_SIDE = 0.5 * ACROSS_FLATS


def _outer_radius(theta: np.ndarray) -> np.ndarray:
    """Distance from the axis to the hexagon boundary at angle ``theta`` (flat normals at multiples of 60 deg)."""
    to_flat_normal = (theta + np.pi / 6) % (np.pi / 3) - np.pi / 6
    return 0.5 * ACROSS_FLATS / np.cos(to_flat_normal)


def _ring_points(num_sectors: int) -> tuple[np.ndarray, np.ndarray]:
    """Inner (bore) and outer (hexagon) boundary points at ``num_sectors`` equal angles, shape ``(n, 2)`` each.

    ``num_sectors`` must be a multiple of 12, so that every hexagon corner is one of the sample angles and the
    outer boundary between two neighbouring samples is a straight piece of one flat.
    """
    assert num_sectors % 12 == 0
    theta = np.arange(num_sectors) * 2 * np.pi / num_sectors
    direction = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return 0.5 * BORE_DIAMETER * direction, _outer_radius(theta)[:, None] * direction


def convex_pieces(num_sectors: int = 12) -> list[np.ndarray]:
    """The nut as ``num_sectors`` convex prisms (one per angular sector), each as an ``(8, 3)`` vertex array.

    Physics engines collide with convex shapes, and a nut is not convex because of its hole. With 12 sectors each
    piece is a quadrilateral prism (bore chord, two radial cuts, one straight piece of a flat). The chords make
    the bore at most 3.4 % narrower than the true circle, about 0.6 mm here.
    """
    inner, outer = _ring_points(num_sectors)
    pieces = []
    for k in range(num_sectors):
        nxt = (k + 1) % num_sectors
        quad = np.stack([inner[k], outer[k], outer[nxt], inner[nxt]])
        bottom = np.hstack([quad, np.full((4, 1), -0.5 * THICKNESS)])
        top = np.hstack([quad, np.full((4, 1), 0.5 * THICKNESS)])
        pieces.append(np.vstack([bottom, top]))
    return pieces


def visual_mesh(num_sectors: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Closed triangle mesh of the nut with a finer bore, ``(vertices (4n, 3), faces (8n, 3))``, outward winding."""
    inner, outer = _ring_points(num_sectors)
    h = 0.5 * THICKNESS
    n = num_sectors
    # vertex blocks: inner bottom, outer bottom, inner top, outer top
    vertices = np.vstack([
        np.hstack([inner, np.full((n, 1), -h)]),
        np.hstack([outer, np.full((n, 1), -h)]),
        np.hstack([inner, np.full((n, 1), h)]),
        np.hstack([outer, np.full((n, 1), h)]),
    ])
    ib, ob, it, ot = 0, n, 2 * n, 3 * n
    faces = []
    for k in range(n):
        j = (k + 1) % n
        faces += [[it + k, ot + k, ot + j], [it + k, ot + j, it + j]]  # top, normal +z
        faces += [[ib + k, ib + j, ob + j], [ib + k, ob + j, ob + k]]  # bottom, normal -z
        faces += [[ob + k, ob + j, ot + j], [ob + k, ot + j, ot + k]]  # outer wall, normal outwards
        faces += [[ib + k, it + k, it + j], [ib + k, it + j, ib + j]]  # bore wall, normal towards the axis
    return vertices, np.array(faces, dtype=np.int64)
