"""Does the threaded nut screw onto the bolt in PhysX? No robot: the nut is spun by an external torque.

Five side-by-side cases, each a bolt standing on the ground with the nut dropped onto its tip:
    tighten       spin clockwise seen from above (right-hand thread: should go down one pitch per turn)
    loosen        spin counter-clockwise (should not go down)
    rest          no torque (should sit on the tip, not fall through)
    tilted        nut starts tilted by 10 deg, then tighten
    phase         nut starts turned by 47 deg, then tighten

    python scripts/test_thread_physics.py --headless

Prints the height and turn count per case, saves the trajectories for ``scripts/render_threads.py``.
"""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=8.0)
parser.add_argument("--spin_rate", type=float, default=0.5, help="Target spin [turns/s].")
parser.add_argument("--output", type=str, default="logs/thread_test.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import math
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from franka_sharpa import nut_geometry, thread_geometry as tg

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    # name, spin direction (-1 = clockwise from above), start tilt [deg], start yaw [deg]
    ("tighten", -1.0, 0.0, 0.0),
    ("loosen", 1.0, 0.0, 0.0),
    ("rest", 0.0, 0.0, 0.0),
    ("tilted", -1.0, 10.0, 0.0),
    ("phase", -1.0, 0.0, 47.0),
]
SPACING = 0.15
SETTLE_S = 0.5
FPS = 30


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args.device))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2000.0))

    bolt_cfg = sim_utils.UsdFileCfg(usd_path=str(ROOT / "assets/nut/bolt.usda"))
    nut_cfg = sim_utils.UsdFileCfg(
        usd_path=str(ROOT / "assets/nut/nut_threaded.usda"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=32, solver_velocity_iteration_count=1),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=nut_geometry.MASS),
    )
    # nut bottom 2 mm above the bolt tip
    start_z = tg.BOLT_LENGTH + 0.5 * nut_geometry.THICKNESS + 0.002
    for i, (_, _, tilt, yaw) in enumerate(CASES):
        x = i * SPACING
        sim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(x, 0.0, 0.0))
        bolt_cfg.func(f"/World/env_{i}/Bolt", bolt_cfg)
        nut_cfg.func(f"/World/env_{i}/Nut", nut_cfg)
    nuts = RigidObject(RigidObjectCfg(prim_path="/World/env_.*/Nut"))
    sim.reset()

    n = len(CASES)
    dev = sim.device
    origins = torch.tensor([[i * SPACING, 0.0, 0.0] for i in range(n)], device=dev)
    tilt = torch.tensor([math.radians(c[2]) for c in CASES], device=dev)
    yaw0 = torch.tensor([math.radians(c[3]) for c in CASES], device=dev)
    quat = quat_from_euler_xyz(tilt, torch.zeros(n, device=dev), yaw0)
    pos = origins + torch.tensor([0.0, 0.0, start_z], device=dev)
    nuts.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
    nuts.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))

    direction = torch.tensor([c[1] for c in CASES], device=dev)
    target_rate = 2 * math.pi * args.spin_rate * direction
    steps = int(args.seconds / sim.get_physics_dt())
    record_every = int(round(1 / (FPS * sim.get_physics_dt())))
    traj_pos, traj_quat = [], []
    turns = torch.zeros(n, device=dev)
    prev_yaw = euler_xyz_from_quat(nuts.data.root_quat_w)[2]
    t0 = time.time()
    for step in range(steps):
        spinning = step * sim.get_physics_dt() > SETTLE_S
        # yaw-rate controller standing in for a hand turning the nut; force-free, so the thread alone sets the height
        rate = nuts.data.root_ang_vel_w[:, 2]
        torque_z = (0.005 * (target_rate - rate)).clamp(-0.03, 0.03) * float(spinning)
        torques = torch.zeros(n, 1, 3, device=dev)
        torques[:, 0, 2] = torque_z
        nuts.set_external_force_and_torque(torch.zeros(n, 1, 3, device=dev), torques, is_global=True)
        nuts.write_data_to_sim()
        sim.step(render=False)
        nuts.update(sim.get_physics_dt())
        yaw = euler_xyz_from_quat(nuts.data.root_quat_w)[2]
        turns += torch.remainder(yaw - prev_yaw + math.pi, 2 * math.pi) - math.pi
        prev_yaw = yaw
        if step % record_every == 0:
            traj_pos.append((nuts.data.root_pos_w - origins).cpu().numpy())
            traj_quat.append(nuts.data.root_quat_w.cpu().numpy())
        if step == int(SETTLE_S / sim.get_physics_dt()):
            settle_z = (nuts.data.root_pos_w[:, 2]).clone()
    wall = time.time() - t0

    z = nuts.data.root_pos_w[:, 2]
    turns = turns / (2 * math.pi)
    print(f"[thread] {steps} physics steps in {wall:.1f} s ({steps / wall:.0f} steps/s for {n} nuts)")
    print(f"[thread] bolt tip at {tg.BOLT_LENGTH * 1e3:.1f} mm; nut centre resting on the tip would be at "
          f"{(tg.BOLT_LENGTH + 0.5 * nut_geometry.THICKNESS) * 1e3:.1f} mm")
    for i, (name, *_ ) in enumerate(CASES):
        dz = (z[i] - settle_z[i]).item()
        expected = -turns[i].item() * tg.PITCH  # clockwise turns are negative
        tilt_now = math.degrees(math.acos(min(1.0, abs(1 - 2 * (nuts.data.root_quat_w[i, 1] ** 2 + nuts.data.root_quat_w[i, 2] ** 2).item()))))
        print(f"[thread] {name:8s} settled at {settle_z[i].item() * 1e3:6.1f} mm, turned {turns[i].item():+5.2f}, "
              f"moved {dz * 1e3:+6.1f} mm (one pitch per turn: {-expected * 1e3 if False else turns[i].item() * tg.PITCH * 1e3:+6.1f} mm), "
              f"axis tilt {tilt_now:4.1f} deg, xy offset {torch.norm(nuts.data.root_pos_w[i, :2] - origins[i, :2]).item() * 1e3:.1f} mm")
    np.savez_compressed(
        ROOT / args.output,
        names=np.array([c[0] for c in CASES]),
        nut_pos=np.stack(traj_pos, axis=1),
        nut_quat=np.stack(traj_quat, axis=1),
        fps=FPS,
    )
    print(f"[thread] saved {args.output}", flush=True)
    import os

    os._exit(0)  # Kit hangs in app.close() after a bare SimulationContext


if __name__ == "__main__":
    main()
    app.close()
