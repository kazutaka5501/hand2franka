"""Write the hex nut as a rigid-body USD, ``assets/nut/nut.usda``.

The file is plain-text USD written directly, so this needs neither Isaac Sim nor pxr:

    python scripts/build_nut.py

One rigid body with a visual mesh (bore smoothed with 48 sectors) and 12 invisible convex collision pieces, see
``franka_sharpa.nut_geometry``.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/nut/nut.usda"

# load the geometry module by path: importing the package would pull in nothing heavy, but keep the script standalone
_spec = importlib.util.spec_from_file_location("nut_geometry", ROOT / "franka_sharpa/nut_geometry.py")
geometry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(geometry)

NUT_COLOR = (0.05, 0.05, 0.05)  # black PLA, like the printed nut


def _points(vertices) -> str:
    return "[" + ", ".join(f"({x:.6f}, {y:.6f}, {z:.6f})" for x, y, z in vertices) + "]"


def _mesh(name: str, vertices, faces, extra: str) -> str:
    counts = ", ".join(str(len(face)) for face in faces)
    indices = ", ".join(str(i) for face in faces for i in face)
    return f"""
    def Mesh "{name}"{extra}
    {{
        int[] faceVertexCounts = [{counts}]
        int[] faceVertexIndices = [{indices}]
        point3f[] points = {_points(vertices)}
        uniform token subdivisionScheme = "none"
"""


def _prism_faces():
    # vertex order of a piece: bottom quad 0-3, top quad 4-7, both counter-clockwise seen from +z
    return [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]


def main():
    vertices, faces = geometry.visual_mesh()
    body = _mesh("visual", vertices, faces, "") + f"""        color3f[] primvars:displayColor = [{NUT_COLOR}]
    }}
"""
    for k, piece in enumerate(geometry.convex_pieces()):
        body += _mesh(
            f"collision_{k:02d}",
            piece,
            _prism_faces(),
            ' (\n        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]\n    )',
        ) + """        uniform token physics:approximation = "convexHull"
        uniform token purpose = "guide"
    }
"""
    text = f"""#usda 1.0
(
    defaultPrim = "Nut"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Nut" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{{
    float physics:mass = {geometry.MASS}
{body}}}
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT}: across flats {geometry.ACROSS_FLATS * 1e3:.0f} mm, thickness {geometry.THICKNESS * 1e3:.0f} mm, "
          f"bore {geometry.BORE_DIAMETER * 1e3:.0f} mm, mass {geometry.MASS * 1e3:.0f} g")


if __name__ == "__main__":
    main()
