"""Pinhole camera + table-frame geometry shared by perception and the simulator.

Table frame: origin at the centre of the A4 reference marker (ArUco id 13),
x to the right of the image, y away from the camera, z up.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

W, H = 1920, 1080
CALIB_PATH = Path("data/calib.json")

ARUCO_DICT = cv2.aruco.DICT_4X4_50
ID_HAND, ID_CUBE_TOP, ID_CUBE_FRONT, ID_TABLE = 0, 1, 2, 13


@dataclass
class Calib:
    f: float  # focal length [px], principal point assumed at the image centre
    rvec: list  # table -> camera
    tvec: list
    table_marker: float  # side of marker 13 [m], measured against the A4 sheet
    cube: float = None  # cube edge [m]
    top_marker: float = None
    front_marker: float = None
    top_offset: list = None  # (x, y) offset of the top marker from the face centre
    front_offset: float = None

    @classmethod
    def load(cls, path=CALIB_PATH):
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path=CALIB_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @property
    def K(self):
        return np.array([[self.f, 0, W / 2], [0, self.f, H / 2], [0, 0, 1.0]])

    @property
    def R(self):
        return cv2.Rodrigues(np.asarray(self.rvec, float))[0]

    @property
    def cam_pos(self):
        return -self.R.T @ np.asarray(self.tvec, float)

    def project(self, pts):
        """Table-frame points (..., 3) -> pixels (..., 2)."""
        pc = np.asarray(pts) @ self.R.T + np.asarray(self.tvec)
        return pc[..., :2] / pc[..., 2:] * self.f + [W / 2, H / 2]

    def rays(self, px):
        """Pixels (N, 2) -> unit ray directions in the table frame."""
        d = np.c_[(np.asarray(px, float) - [W / 2, H / 2]) / self.f, np.ones(len(px))] @ self.R
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def on_plane(self, px, z):
        """Intersect pixel rays with the horizontal plane at height z."""
        d = self.rays(px)
        return self.cam_pos + d * ((z - self.cam_pos[2]) / d[:, 2])[:, None]

    def cam_to_table(self, pc):
        return (np.asarray(pc) - np.asarray(self.tvec)) @ self.R


def square(s):
    """Corners of a marker of side s in OpenCV ArUco order (TL, TR, BR, BL), z out of the marker."""
    return np.array([[-s / 2, s / 2, 0], [s / 2, s / 2, 0], [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]])


def cube_marker_points(cube, top_marker, front_marker, top_offset=(0, 0), front_offset=0.0):
    """Top- and front-face marker corners in the cube frame (origin on the table under the centre)."""
    top = square(top_marker) + [top_offset[0], top_offset[1], cube]
    b = (cube - front_marker) / 2
    front = square(front_marker)[:, [0, 2, 1]] + [front_offset, -cube / 2, b + front_marker / 2]
    return np.r_[top, front]


def aruco_detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), params)


def detect(detector, bgr):
    """-> {marker id: (4, 2) corners}"""
    corners, ids, _ = detector.detectMarkers(bgr)
    return {} if ids is None else {int(i): c[0] for c, i in zip(corners, ids.ravel())}
