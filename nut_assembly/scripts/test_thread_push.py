"""How hard must the nut be pushed straight down before it goes through the thread without turning?

A real nut cannot be pushed down a thread; in simulation the SDF contact gives way at some force, and the stage-2
policies found that (the Franka pressed the nut 16-26 mm down with almost no rotation). Here the nut rests on the bolt
tip and a constant downward force is applied for 3 s, no torque, one force per bolt:

    python scripts/test_thread_push.py --headless --suffix _c10     # assets/nut/{bolt,nut_threaded}_c10.usda

Prints, per force, how far the nut went down and how much it turned. A thread that holds gives ~0 mm (a little
settling); anything beyond a few millimetres without a matching turn is the nut passing through the thread.
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--suffix", type=str, default="")
parser.add_argument("--forces", type=float, nargs="+", default=[5, 10, 20, 50, 100, 200])
parser.add_argument("--seconds", type=float, default=3.0)
parser.add_argument("--sdf_resolution", type=int, default=None, help="Override the SDF resolution of both meshes.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.utils.math import euler_xyz_from_quat

from franka_sharpa import nut_geometry, thread_geometry as tg

ROOT = Path(__file__).resolve().parents[1]
SPACING = 0.15
SETTLE_S = 0.5


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args.device))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    bolt_path, nut_path = ROOT / f"assets/nut/bolt{args.suffix}.usda", ROOT / f"assets/nut/nut_threaded{args.suffix}.usda"
    if args.sdf_resolution:
        for p in (bolt_path, nut_path):
            text = p.read_text().replace("sdfResolution = 256", f"sdfResolution = {args.sdf_resolution}")
            q = p.with_name(p.stem + f"_sdf{args.sdf_resolution}.usda")
            q.write_text(text)
        bolt_path = bolt_path.with_name(bolt_path.stem + f"_sdf{args.sdf_resolution}.usda")
        nut_path = nut_path.with_name(nut_path.stem + f"_sdf{args.sdf_resolution}.usda")
    bolt_cfg = sim_utils.UsdFileCfg(usd_path=str(bolt_path))
    nut_cfg = sim_utils.UsdFileCfg(
        usd_path=str(nut_path),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=32, solver_velocity_iteration_count=1),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=nut_geometry.MASS),
    )
    n = len(args.forces)
    for i in range(n):
        sim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(i * SPACING, 0.0, 0.0))
        bolt_cfg.func(f"/World/env_{i}/Bolt", bolt_cfg)
        nut_cfg.func(f"/World/env_{i}/Nut", nut_cfg)
    nuts = RigidObject(RigidObjectCfg(prim_path="/World/env_.*/Nut"))
    sim.reset()
    dev = sim.device
    origins = torch.tensor([[i * SPACING, 0.0, 0.0] for i in range(n)], device=dev)
    pos = origins + torch.tensor([0.0, 0.0, tg.BOLT_LENGTH + 0.5 * nut_geometry.THICKNESS + 0.002], device=dev)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev).repeat(n, 1)
    nuts.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
    nuts.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
    force = torch.tensor(args.forces, device=dev)

    dt = sim.get_physics_dt()
    settle_steps = int(SETTLE_S / dt)
    turns = torch.zeros(n, device=dev)
    prev_yaw = euler_xyz_from_quat(nuts.data.root_quat_w)[2]
    for step in range(settle_steps + int(args.seconds / dt)):
        f = torch.zeros(n, 1, 3, device=dev)
        if step >= settle_steps:
            f[:, 0, 2] = -force
        nuts.set_external_force_and_torque(f, torch.zeros(n, 1, 3, device=dev), is_global=True)
        nuts.write_data_to_sim()
        sim.step(render=False)
        nuts.update(dt)
        yaw = euler_xyz_from_quat(nuts.data.root_quat_w)[2]
        turns += torch.remainder(yaw - prev_yaw + math.pi, 2 * math.pi) - math.pi
        prev_yaw = yaw
        if step == settle_steps:
            z0 = nuts.data.root_pos_w[:, 2].clone()
            turns.zero_()
    dz = 1e3 * (z0 - nuts.data.root_pos_w[:, 2])
    turns = -turns / (2 * math.pi)
    print(f"[push] clearance {tg.CLEARANCE * 1e3:.1f} mm (asset {args.suffix or 'default'}), sdf {args.sdf_resolution or 256}")
    for i, F in enumerate(args.forces):
        held = "holds" if dz[i] < 3.0 else "PUSHED THROUGH" if dz[i] > abs(turns[i].item()) * 6.0 + 3.0 else "screwed"
        print(f"[push]   {F:6.0f} N: down {dz[i].item():6.1f} mm, turned {turns[i].item():+.2f}  -> {held}", flush=True)
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
