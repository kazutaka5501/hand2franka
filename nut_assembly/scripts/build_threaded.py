"""Write the threaded bolt and nut for the screwing stage as plain-text USD (no Isaac Sim needed):

    python scripts/build_threaded.py

    assets/nut/nut_threaded.usda   rigid body, one SDF collision mesh (hex body + internal thread)
    assets/nut/bolt.usda           static collider: convex hex head + SDF threaded shank

Thread meshes and SDF collision as in Play2Perfect's screwing task (thread as SDF, resolution 256) and Isaac Lab's
Factory nut-and-bolt task. Friction 0.3, roughly PLA on PLA. See ``franka_sharpa.thread_geometry``.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from franka_sharpa import nut_geometry, thread_geometry as tg  # noqa: E402

OUT_DIR = ROOT / "assets/nut"
SDF_RESOLUTION = 256
FRICTION = 0.3


def _points(vertices) -> str:
    return "[" + ", ".join(f"({x:.7f}, {y:.7f}, {z:.7f})" for x, y, z in vertices) + "]"


def _mesh(name, vertices, faces, schemas=(), attributes="", material=None) -> str:
    counts = ", ".join(str(len(face)) for face in faces)
    indices = ", ".join(str(i) for face in faces for i in face)
    meta = ""
    if schemas:
        meta = " (\n        prepend apiSchemas = [" + ", ".join(f'"{s}"' for s in schemas) + "]\n    )"
    binding = f"        rel material:binding:physics = <{material}>\n" if material else ""
    return f"""
    def Mesh "{name}"{meta}
    {{
        int[] faceVertexCounts = [{counts}]
        int[] faceVertexIndices = [{indices}]
        point3f[] points = {_points(vertices)}
        uniform token subdivisionScheme = "none"
{attributes}{binding}    }}
"""


def _material(root: str) -> str:
    return f"""
    def Material "PhysicsMaterial" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    )
    {{
        float physics:staticFriction = {FRICTION}
        float physics:dynamicFriction = {FRICTION}
        float physics:restitution = 0
    }}
"""


SDF_SCHEMAS = ("PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxSDFMeshCollisionAPI", "MaterialBindingAPI")
SDF_ATTRIBUTES = f"""        uniform token physics:approximation = "sdf"
        int physxSDFMeshCollision:sdfResolution = {SDF_RESOLUTION}
"""
HULL_SCHEMAS = ("PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI")
HULL_ATTRIBUTES = '        uniform token physics:approximation = "convexHull"\n'


def _color(rgb) -> str:
    return f"        color3f[] primvars:displayColor = [{tuple(rgb)}]\n"


def _file(root: str, body: str, root_schemas=(), root_attributes="") -> str:
    meta = ""
    if root_schemas:
        meta = " (\n    prepend apiSchemas = [" + ", ".join(f'"{s}"' for s in root_schemas) + "]\n)"
    return f"""#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}"{meta}
{{
{root_attributes}{body}}}
"""


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clearance", type=float, default=None, help="Radial thread clearance [m] (default: thread_geometry).")
    parser.add_argument("--suffix", type=str, default="", help="Appended to the file names, e.g. _c15.")
    args = parser.parse_args()
    if args.clearance is not None:
        tg.CLEARANCE = args.clearance
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vertices, faces = tg.nut_mesh()
    body = _material("Nut") + _mesh(
        "collision", vertices, faces, SDF_SCHEMAS, SDF_ATTRIBUTES + _color((0.05, 0.05, 0.05)), "/Nut/PhysicsMaterial"
    )
    (OUT_DIR / f"nut_threaded{args.suffix}.usda").write_text(
        _file("Nut", body, ("PhysicsRigidBodyAPI", "PhysicsMassAPI"), f"    float physics:mass = {nut_geometry.MASS}\n")
    )

    shank_v, shank_f = tg.bolt_shank_mesh()
    body = _material("Bolt")
    body += _mesh("head", tg.bolt_head_vertices(), tg.bolt_head_faces(), HULL_SCHEMAS,
                  HULL_ATTRIBUTES + _color((0.1, 0.2, 0.8)), "/Bolt/PhysicsMaterial")
    body += _mesh("shank", shank_v, shank_f, SDF_SCHEMAS, SDF_ATTRIBUTES + _color((0.1, 0.2, 0.8)), "/Bolt/PhysicsMaterial")
    (OUT_DIR / f"bolt{args.suffix}.usda").write_text(_file("Bolt", body))

    print(f"wrote {OUT_DIR}/nut_threaded.usda and bolt.usda: M{tg.MAJOR_DIAMETER * 1e3:.0f}x{tg.PITCH * 1e3:.0f}, "
          f"minor {tg.MINOR_DIAMETER * 1e3:.1f} mm, clearance {tg.CLEARANCE * 1e3:.1f} mm, "
          f"bolt length {tg.BOLT_LENGTH * 1e3:.0f} mm")


if __name__ == "__main__":
    main()
