"""Build the Franka Panda arm + Sharpa Wave right hand robot.

Step 1 merges the two URDFs (hand flange bolted onto ``panda_link8``) and adds a ``palm`` frame.
Step 2 converts the merged URDF to USD with Isaac Lab's URDF converter.

    python scripts/build_robot.py            # both steps
    python scripts/build_robot.py --urdf-only  # no simulator needed
"""

import argparse
import copy
import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARPA_DIR = ROOT / "third_party/sharpa-urdf-usd-xml/wave_01/right_sharpa_wave"
SHARPA_URDF = SHARPA_DIR / "right_sharpa_wave_with_flange.urdf"
OUT_DIR = ROOT / "assets/franka_sharpa"
OUT_URDF = OUT_DIR / "franka_sharpa.urdf"

ARM_FLANGE_LINK = "panda_link8"
HAND_ROOT_LINK = "right_hand_flange"
HAND_BASE_LINK = "right_hand_C_MC"
# Rotation of the hand about the flange axis. With this value the palm faces down when the arm reaches forward
# in its usual elbow-up posture.
MOUNT_RPY = "0 0 2.3561944902"
# Palm frame: centre of the palm surface, expressed in the hand base link (z towards the fingers, x out of the palm).
PALM_XYZ = "0.02 0 0.07"


def franka_description_dir() -> Path:
    # located without importing isaacsim, which would start the Omniverse Kit bootstrap
    exts = Path(importlib.util.find_spec("isaacsim").origin).parent / "exts"
    return exts / "isaacsim.asset.importer.urdf/data/urdf/robots/franka_description"


def absolutize_meshes(robot: ET.Element, package: str, package_dir: Path):
    prefix = f"package://{package}/"
    for mesh in robot.iter("mesh"):
        filename = mesh.get("filename")
        if filename.startswith(prefix):
            mesh.set("filename", str(package_dir / filename[len(prefix) :]))


def frame_link(name: str) -> ET.Element:
    """A body with negligible inertia and no geometry, used as a reference frame."""
    link = ET.Element("link", name=name)
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value="1.0E-06")
    ET.SubElement(inertial, "inertia", ixx="1.0E-10", ixy="0", ixz="0", iyy="1.0E-10", iyz="0", izz="1.0E-10")
    return link


def fixed_joint(name: str, parent: str, child: str, xyz: str, rpy: str) -> ET.Element:
    joint = ET.Element("joint", name=name, type="fixed")
    ET.SubElement(joint, "origin", xyz=xyz, rpy=rpy)
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    return joint


def merge_urdf() -> Path:
    franka_dir = franka_description_dir()
    arm = ET.parse(franka_dir / "robots/panda_arm.urdf").getroot()
    hand = ET.parse(SHARPA_URDF).getroot()
    absolutize_meshes(arm, "franka_description", franka_dir)
    absolutize_meshes(hand, "right_sharpa_wave", SHARPA_DIR)

    robot = ET.Element("robot", name="franka_sharpa")
    for element in arm:
        if element.tag in ("link", "joint"):
            robot.append(copy.deepcopy(element))
    robot.append(fixed_joint("hand_mount_joint", ARM_FLANGE_LINK, HAND_ROOT_LINK, "0 0 0", MOUNT_RPY))
    for element in hand:
        if element.tag in ("link", "joint"):
            robot.append(copy.deepcopy(element))
    robot.append(frame_link("palm"))
    robot.append(fixed_joint("palm_fix_joint", HAND_BASE_LINK, "palm", PALM_XYZ, "0 0 0"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ET.indent(robot)
    ET.ElementTree(robot).write(OUT_URDF, encoding="utf-8", xml_declaration=True)
    links, joints = robot.findall("link"), robot.findall("joint")
    moving = [j for j in joints if j.get("type") != "fixed"]
    print(f"[build_robot] wrote {OUT_URDF}: {len(links)} links, {len(joints)} joints ({len(moving)} actuated)")
    return OUT_URDF


def convert_to_usd(urdf_path: Path):
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        usd_dir=str(OUT_DIR),
        usd_file_name="franka_sharpa.usd",
        force_usd_conversion=True,
        fix_base=True,
        # keep the fingertip, pad and palm links as separate bodies
        merge_fixed_joints=False,
        self_collision=True,
        collider_type="convex_hull",
        # gains are placeholders, the actuator models in franka_sharpa/robot.py override them at runtime
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=1.0),
        ),
    )
    converter = UrdfConverter(cfg)
    print(f"[build_robot] wrote {converter.usd_path}")
    app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf-only", action="store_true", help="Only merge the URDFs, skip the USD conversion.")
    args = parser.parse_args()
    path = merge_urdf()
    if not args.urdf_only:
        convert_to_usd(path)
