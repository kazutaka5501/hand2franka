"""Write the threaded nut and bolt in the layout Isaac Lab's Factory NutThread task expects (no Isaac Sim needed):

    python scripts/build_factory_assets.py

    assets/nut/nut_factory.usda    held asset: rigid body + articulation root (no joints), SDF thread
    assets/nut/bolt_factory.usda   fixed asset: rigid body fixed to the world by an articulation-root fixed joint

Same meshes, SDF settings and material as ``build_threaded.py`` (``nut_threaded.usda``, ``bolt.usda``, which are
left as they are). What Factory needs differently, copied from its factory_nut_m16.usd / factory_bolt_m16.usd:

    * both parts are loaded as ``Articulation`` (``actuators={}``): the rigid body is a child Xform of the default
      prim; the nut body carries ``ArticulationRootAPI`` itself, the bolt has a ``PhysicsFixedJoint "root_joint"``
      with ``ArticulationRootAPI`` and only ``body1`` set (fixed-base articulation, pose still settable on reset)
    * bolt frame: origin at the centre of the head's underside, z up (same as ``bolt.usda``)
    * nut frame: NOT at the nut centre. Factory's ``get_held_base_pos_local`` puts the nut's "base" (its bottom
      face) at +``base_height`` (the bolt head height) in the nut frame, i.e. the nut frame coincides with the bolt
      frame once the nut is screwed all the way down onto the head. The M16 nut mesh spans z = 0.010 .. 0.023 for
      a 0.010 head; here the nut spans z = 0.025 .. 0.058.
    * hex flats face +-x in both (as in the M16 nut), so Factory's ``held_asset_rot_init = -90`` deg still puts
      the flats against the fingers.

Thread phase: the nut mates with the bolt (zero relative yaw) when its frame sits at z = k P - 0.0415 on the bolt
axis, i.e. z = 0.0005 m mod 6 mm.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from franka_sharpa import nut_geometry, thread_geometry as tg  # noqa: E402
from build_threaded import (  # noqa: E402
    HULL_ATTRIBUTES,
    HULL_SCHEMAS,
    OUT_DIR,
    SDF_ATTRIBUTES,
    SDF_SCHEMAS,
    _color,
    _file,
    _material,
    _mesh,
)

# offset from the nut frame (Factory convention) to the nut centre
NUT_FRAME_TO_CENTRE = tg.BOLT_HEAD_HEIGHT + 0.5 * nut_geometry.THICKNESS
BOLT_MASS = 0.05  # irrelevant (fixed base), Factory's default

ARTICULATION_ATTRIBUTES = """        bool physxArticulation:enabledSelfCollisions = 0
        int physxArticulation:solverPositionIterationCount = 32
        int physxArticulation:solverVelocityIterationCount = 1
"""


def _body(name: str, schemas, attributes: str, children: str) -> str:
    meta = " (\n        prepend apiSchemas = [" + ", ".join(f'"{s}"' for s in schemas) + "]\n    )"
    return f"""
    def Xform "{name}"{meta}
    {{
{attributes}{children}    }}
"""


def nut_factory() -> str:
    vertices, faces = tg.nut_mesh()
    vertices = vertices + np.array([0.0, 0.0, NUT_FRAME_TO_CENTRE])
    children = _material("Nut") + _mesh(
        "collisions", vertices, faces, SDF_SCHEMAS, SDF_ATTRIBUTES + _color((0.05, 0.05, 0.05)),
        "/NutFactory/nut/PhysicsMaterial",
    )
    body = _body(
        "nut",
        ("PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsArticulationRootAPI", "PhysxArticulationAPI"),
        f"        float physics:mass = {nut_geometry.MASS}\n" + ARTICULATION_ATTRIBUTES,
        children,
    )
    return _file("NutFactory", body)


def bolt_factory() -> str:
    shank_v, shank_f = tg.bolt_shank_mesh()
    material = "/BoltFactory/bolt/PhysicsMaterial"
    children = _material("Bolt")
    children += _mesh("head", tg.bolt_head_vertices(), tg.bolt_head_faces(), HULL_SCHEMAS,
                      HULL_ATTRIBUTES + _color((0.1, 0.2, 0.8)), material)
    children += _mesh("shank", shank_v, shank_f, SDF_SCHEMAS, SDF_ATTRIBUTES + _color((0.1, 0.2, 0.8)), material)
    body = _body("bolt", ("PhysicsRigidBodyAPI", "PhysicsMassAPI"), f"        float physics:mass = {BOLT_MASS}\n", children)
    body += f"""
    def PhysicsFixedJoint "root_joint" (
        prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]
    )
    {{
        rel physics:body1 = </BoltFactory/bolt>
{ARTICULATION_ATTRIBUTES}    }}
"""
    return _file("BoltFactory", body)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "nut_factory.usda").write_text(nut_factory())
    (OUT_DIR / "bolt_factory.usda").write_text(bolt_factory())
    print(f"wrote {OUT_DIR}/nut_factory.usda (nut z = {tg.BOLT_HEAD_HEIGHT:.3f} .. "
          f"{tg.BOLT_HEAD_HEIGHT + nut_geometry.THICKNESS:.3f} in its frame) and bolt_factory.usda "
          f"(tip at z = {tg.BOLT_LENGTH:.3f}); clearance {tg.CLEARANCE * 1e3:.1f} mm")


if __name__ == "__main__":
    main()
