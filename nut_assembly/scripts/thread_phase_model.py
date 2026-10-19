"""How much clockwise turning the Factory nut-thread task needs, from the thread geometry alone (numpy, no Isaac Sim).

    python scripts/thread_phase_model.py

Model: the nut is held coaxial and upright (as Factory's gripper holds it) and pushed down. At each relative yaw
phi = nut yaw - bolt yaw it sits as low as the meshes allow without overlap (nut bore radius >= bolt radius
everywhere), riding up a flank if turning requires it. The nut first drops straight onto the bolt tip, then turns
clockwise (phi decreasing) in 2 deg steps.

Result for the as-built parts: lowered straight down, the nut's 45 deg bore countersink sits on the bolt's 45 deg tip
chamfer with its bottom face 4.7 mm below the tip (the Isaac smoke test measured 4.75 mm) for most phases; the
thread only catches near phi = 5..10 deg. From a start phase phi0 the nut reaches Factory's success depth (bottom
1.125 pitch = 6.75 mm below the tip) after about phi0 + 44 deg of clockwise turning (phi0 in 10..360 deg).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from franka_sharpa import nut_geometry as ng, thread_geometry as tg  # noqa: E402

P, TIP, H = tg.PITCH, tg.BOLT_LENGTH, 0.5 * ng.THICKNESS
COUNTERSINK = 0.0005  # as in thread_geometry.nut_mesh: bore opens to major/2 + clearance + 0.5 mm at the faces
SUCCESS_DEPTH = (1.5 - 0.375) * P  # Factory: base within 0.375 pitch of (tip - 1.5 pitch)
THETA = np.linspace(0, 2 * np.pi, 240, endpoint=False)[:, None]
DEPTHS = np.arange(-0.002, 0.012, 0.0001)  # nut bottom face below the tip


def bolt_r(z):
    return np.minimum(tg.profile(z - P * THETA / (2 * np.pi)), 0.5 * tg.MINOR_DIAMETER + (TIP - z))


def nut_r(z, zc, phi):
    r = tg.profile((z - zc) - P * (THETA - phi) / (2 * np.pi)) + tg.CLEARANCE
    return np.maximum(r, 0.5 * tg.MAJOR_DIAMETER + tg.CLEARANCE + COUNTERSINK - (H - np.abs(z - zc)))


_cache = {}


def free(phi_deg: float, j: int) -> bool:
    key = (round(phi_deg % 360, 3), j)
    if key not in _cache:
        zc = TIP - DEPTHS[j] + H
        z = np.arange(zc - H, min(zc + H, TIP), 0.0001)[None, :]
        _cache[key] = z.size == 0 or not np.any(nut_r(z, zc, np.deg2rad(phi_deg)) < bolt_r(z) - 1e-5)
    return _cache[key]


def settle(phi: float, j: int) -> int:
    while not free(phi, j):
        j -= 1
    while j + 1 < len(DEPTHS) and free(phi, j + 1):
        j += 1
    return j


def main():
    print(f"success depth {1e3 * SUCCESS_DEPTH:.2f} mm below the tip")
    for phi0 in range(0, 360, 10):
        j = settle(phi0, 0)
        seat, phi, rot = DEPTHS[j], phi0, 0
        while DEPTHS[j] < SUCCESS_DEPTH and rot < 400:
            phi, rot = phi - 2, rot + 2
            j = settle(phi, j)
        need = f"{rot} deg" if rot < 400 else "> 400 deg"
        print(f"phi0 {phi0:4d} deg: seat {1e3 * seat:5.2f} mm below tip, clockwise turn to success {need}")


if __name__ == "__main__":
    main()
