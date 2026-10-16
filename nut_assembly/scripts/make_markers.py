"""Printable ArUco markers for recording the nut demonstrations, and their 3D layout.

Writes to ``assets/markers/``:
    markers.pdf    page 1: table board (A4, ids 10-17, bolt position in the centre)
                   page 2: six nut markers (ids 0-5, 22 mm) plus a spare set, with cut lines
    layout.json    corner coordinates of every marker: table ids in the board frame, nut ids in the nut frame

Frames (metres):
    board  origin at the sheet centre (= bolt axis), x to the right of the sheet, y towards its top edge, z up
    nut    as ``franka_sharpa.nut_geometry``: origin at the centre, z along the axis towards the face marked "T",
           side-face normals at 0, 60, ..., 300 deg; marker k sits on the face with normal at 60k deg, so seen from
           the "T" face the ids run counter-clockwise, and each marker's top edge points towards "T"
Corner order is ArUco's: top-left, top-right, bottom-right, bottom-left of the marker as printed.

Print at 100 % ("actual size", no fit-to-page) and check the 100 mm scale bars with a ruler.
Needs OpenCV with aruco and Pillow (e.g. the hand2franka environment):

    python scripts/make_markers.py
"""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/markers"
_spec = importlib.util.spec_from_file_location("nut_geometry", ROOT / "franka_sharpa/nut_geometry.py")
nut = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nut)

DICT = cv2.aruco.DICT_4X4_50
DPI = 600
A4_MM = (210.0, 297.0)

TABLE_IDS = list(range(10, 18))
TABLE_MARKER_MM = 40.0
NUT_IDS = list(range(6))
NUT_MARKER_MM = 22.0
SIDE_FACE_MM = 1e3 * nut.ACROSS_FLATS / np.sqrt(3)  # width of one side face, 29.4 mm
THICKNESS_MM = 1e3 * nut.THICKNESS


def px(mm: float) -> int:
    return int(round(mm * DPI / 25.4))


def marker_image(marker_id: int, side_mm: float) -> Image.Image:
    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    return Image.fromarray(cv2.aruco.generateImageMarker(dictionary, marker_id, px(side_mm)))


def font(size_mm: float):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, px(size_mm))
    return ImageFont.load_default()


class Page:
    """A4 portrait canvas addressed in millimetres from the top-left corner."""

    def __init__(self):
        self.image = Image.new("L", (px(A4_MM[0]), px(A4_MM[1])), 255)
        self.draw = ImageDraw.Draw(self.image)

    def paste_centered(self, img: Image.Image, cx_mm: float, cy_mm: float):
        self.image.paste(img, (px(cx_mm) - img.width // 2, px(cy_mm) - img.height // 2))

    def text(self, x_mm, y_mm, s, size_mm=3.5, anchor="mm"):
        self.draw.text((px(x_mm), px(y_mm)), s, fill=0, font=font(size_mm), anchor=anchor)

    def line(self, pts_mm, width_mm=0.3, fill=0):
        self.draw.line([(px(x), px(y)) for x, y in pts_mm], fill=fill, width=max(1, px(width_mm)))

    def circle(self, cx, cy, r, width_mm=0.3, fill=0):
        self.draw.ellipse([px(cx - r), px(cy - r), px(cx + r), px(cy + r)], outline=fill, width=max(1, px(width_mm)))

    def scale_bar(self, x_mm, y_mm):
        self.line([(x_mm, y_mm), (x_mm + 100, y_mm)], 0.4)
        for k in range(11):
            tick = 3 if k % 5 == 0 else 1.5
            self.line([(x_mm + 10 * k, y_mm - tick), (x_mm + 10 * k, y_mm)], 0.3)
        self.text(x_mm + 50, y_mm + 3.5, "100 mm - check with a ruler", 2.8)


def table_page(layout: dict) -> Image.Image:
    page = Page()
    w, h = A4_MM
    s = TABLE_MARKER_MM
    # marker centres in the board frame [mm]: corners, edge midpoints left/right and top/bottom
    cx, cy = 0.5 * w - 15 - 0.5 * s, 0.5 * h - 15 - 0.5 * s
    centres = [(-cx, cy), (0, cy), (cx, cy), (cx, 0), (cx, -cy), (0, -cy), (-cx, -cy), (-cx, 0)]
    for marker_id, (x, y) in zip(TABLE_IDS, centres):
        page.paste_centered(marker_image(marker_id, s), 0.5 * w + x, 0.5 * h - y)
        page.text(0.5 * w + x, 0.5 * h - y + 0.5 * s + 3, f"id {marker_id}", 2.5)
        corners = [(x - s / 2, y + s / 2), (x + s / 2, y + s / 2), (x + s / 2, y - s / 2), (x - s / 2, y - s / 2)]
        layout["table"][str(marker_id)] = [[1e-3 * a, 1e-3 * b, 0.0] for a, b in corners]
    # bolt position: crosshair and rings at the board origin
    ox, oy = 0.5 * w, 0.5 * h
    page.line([(ox - 45, oy), (ox + 45, oy)], 0.3)
    page.line([(ox, oy - 45), (ox, oy + 45)], 0.3)
    for r in (20, 30, 35):
        page.circle(ox, oy, r, 0.25)
    page.text(ox, oy - 40, "BOLT: centre the head here", 3)
    page.text(ox, oy + 52, "this edge faces the CAMERA", 3.5)
    page.line([(ox - 10, oy + 57), (ox, oy + 62), (ox + 10, oy + 57)], 0.5)
    page.scale_bar(0.5 * w - 50, h - 6)
    page.text(0.5 * w, 6, "TABLE BOARD  (ArUco 4x4_50, ids 10-17, 40 mm)  - tape flat, do not move during recording", 3)
    return page.image


def nut_page(layout: dict) -> Image.Image:
    page = Page()
    w, _ = A4_MM
    s = NUT_MARKER_MM
    box_w, box_h = SIDE_FACE_MM - 1.5, THICKNESS_MM - 2.0  # cut a little inside the face so the label lies flat
    for row, label in enumerate(("SET A", "SET B (spare)")):
        y0 = 40 + row * 105
        page.text(15, y0 - 18, label, 4, anchor="lm")
        for k, marker_id in enumerate(NUT_IDS):
            x = 45 + (k % 3) * 60
            y = y0 + (k // 3) * 48 + 0.5 * box_h
            page.paste_centered(marker_image(marker_id, s), x, y)
            # dashed cut box
            for x1, y1, x2, y2 in (
                (x - box_w / 2, y - box_h / 2, x + box_w / 2, y - box_h / 2),
                (x + box_w / 2, y - box_h / 2, x + box_w / 2, y + box_h / 2),
                (x + box_w / 2, y + box_h / 2, x - box_w / 2, y + box_h / 2),
                (x - box_w / 2, y + box_h / 2, x - box_w / 2, y - box_h / 2),
            ):
                n = 12
                for i in range(0, n, 2):
                    a, b = i / n, (i + 1) / n
                    page.line([(x1 + a * (x2 - x1), y1 + a * (y2 - y1)), (x1 + b * (x2 - x1), y1 + b * (y2 - y1))], 0.15, 120)
            page.text(x, y - box_h / 2 - 7, "T", 4)
            page.line([(x, y - box_h / 2 - 3), (x, y - box_h / 2 - 1.5)], 0.4)
            page.text(x, y + box_h / 2 + 4, f"id {marker_id}", 3)
    page.text(15, 255, "1. Mark one hex face of the nut with a pen: this is the T face.", 3.2, anchor="lm")
    page.text(15, 261, "2. Cut out ids 0-5 along the dashed lines and stick one on each side face,", 3.2, anchor="lm")
    page.text(15, 266, "   centred, the T arrow pointing to the T face.", 3.2, anchor="lm")
    page.text(15, 272, "3. Seen from the T face, ids 0,1,2,3,4,5 go counter-clockwise.", 3.2, anchor="lm")
    page.scale_bar(0.5 * w - 50, 288)
    page.text(0.5 * w, 10, f"NUT MARKERS  (ArUco 4x4_50, ids 0-5, {s:.0f} mm)", 4)

    # 3D corners on the nut, nut frame
    half = 0.5e-3 * s
    for k, marker_id in enumerate(NUT_IDS):
        theta = np.deg2rad(60 * k)
        normal = np.array([np.cos(theta), np.sin(theta), 0.0])
        right = np.array([-np.sin(theta), np.cos(theta), 0.0])  # seen from outside with T up
        up = np.array([0.0, 0.0, 1.0])
        centre = 0.5 * nut.ACROSS_FLATS * normal
        corners = [centre + half * (-right + up), centre + half * (right + up), centre + half * (right - up), centre + half * (-right - up)]
        layout["nut"][str(marker_id)] = [c.round(6).tolist() for c in corners]
    return page.image


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    layout = {
        "dictionary": "DICT_4X4_50",
        "table_marker_size": 1e-3 * TABLE_MARKER_MM,
        "nut_marker_size": 1e-3 * NUT_MARKER_MM,
        "bolt_axis_in_board_frame": [0.0, 0.0],
        "table": {},
        "nut": {},
    }
    pages = [table_page(layout), nut_page(layout)]
    pages[0].save(OUT / "markers.pdf", save_all=True, append_images=pages[1:], resolution=DPI)
    for i, page in enumerate(pages):
        page.resize((page.width // 6, page.height // 6)).save(OUT / f"preview_page{i + 1}.png")
    (OUT / "layout.json").write_text(json.dumps(layout, indent=1))
    print(f"wrote {OUT / 'markers.pdf'} and layout.json ({len(layout['table'])} table, {len(layout['nut'])} nut markers)")


if __name__ == "__main__":
    main()
