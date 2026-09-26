"""Scene layout of the fixed camera and per-video registration.

All geometry (lanes, stop line, crosswalks, signal heads, the learned flow
field) is stored once, in the coordinates of a reference frame (the median
background of sample C3896 at 1920x1080). The camera is re-mounted between
recordings, so each video is registered to the reference with a homography
(SIFT + RANSAC on its median background) and all track coordinates are mapped
into reference space before any rule runs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

from . import config

MIN_INLIERS = 40          # fewer SIFT inliers -> registration is not trusted
MAX_CORNER_SHIFT = 250.0  # px; a larger implied shift means the fit went wrong


@lru_cache(maxsize=1)
def load_layout() -> dict:
    return json.loads(config.SCENE_FILE.read_text())


BANK_FILE = config.ROOT / "assets" / "bank" / "bank.json"


@lru_cache(maxsize=1)
def _bank() -> list[tuple[np.ndarray, tuple, np.ndarray]]:
    """(H_bank_to_reference, keypoints, descriptors) for every lighting reference."""
    sift = cv2.SIFT_create(4000)
    out = []
    for item in json.loads(BANK_FILE.read_text())["images"].values():
        img = cv2.imread(str(BANK_FILE.parent.parent / item["file"]), cv2.IMREAD_GRAYSCALE)
        kp, desc = sift.detectAndCompute(img, None)
        out.append((np.array(item["H_to_reference"]), kp, desc))
    return out


def _fit(kp, desc, ref_kp, ref_desc) -> tuple[np.ndarray | None, int]:
    matches = cv2.BFMatcher().knnMatch(desc, ref_desc, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.7 * n.distance]
    if len(good) < MIN_INLIERS:
        return None, len(good)
    src = np.float32([kp[m.queryIdx].pt for m in good])
    dst = np.float32([ref_kp[m.trainIdx].pt for m in good])
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return H, int(inliers.sum()) if inliers is not None else 0


GOOD_ENOUGH_INLIERS = 300   # stop trying other bank images once a fit this strong is found


def register(frame_bgr: np.ndarray, prefer: int = 0) -> tuple[np.ndarray, int, bool]:
    """Homography mapping analysis-frame pixels -> reference pixels.

    The frame is matched against each lighting reference of the bank (noon,
    evening sun, dusk); the fit with most RANSAC inliers is composed with that
    reference's homography. Returns (H, n_inliers, ok); when no fit is strong
    enough (or it implies an implausible camera shift) ok is False and H is
    the identity - callers then keep their previous estimate.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if gray.shape[::-1] != config.ANALYSIS_SIZE:
        gray = cv2.resize(gray, config.ANALYSIS_SIZE)
    kp, desc = cv2.SIFT_create(4000).detectAndCompute(gray, None)
    if desc is None or len(kp) < MIN_INLIERS:
        return np.eye(3), 0, False
    bank = _bank()
    best_H, best_n = None, 0
    order = [prefer % len(bank)] + [i for i in range(len(bank)) if i != prefer % len(bank)]
    for i in order:
        H_bank, ref_kp, ref_desc = bank[i]
        H, n = _fit(kp, desc, ref_kp, ref_desc)
        if H is not None and n > best_n:
            best_H, best_n, register.last_bank = H_bank @ H, n, i
        if best_n >= GOOD_ENOUGH_INLIERS:
            break
    if best_H is None or best_n < MIN_INLIERS:
        return np.eye(3), best_n, False
    w, h = config.ANALYSIS_SIZE
    corners = np.float32([[0, 0], [w, 0], [0, h], [w, h]]).reshape(-1, 1, 2)
    if np.abs(cv2.perspectiveTransform(corners, best_H) - corners).max() > MAX_CORNER_SHIFT:
        return np.eye(3), best_n, False
    return best_H / best_H[2, 2], best_n, True


register.last_bank = 0


def corner_shift(H1: np.ndarray, H2: np.ndarray) -> float:
    """Largest displacement (px) of the frame corners between two registrations."""
    w, h = config.ANALYSIS_SIZE
    c = np.float32([[0, 0], [w, 0], [0, h], [w, h], [w / 2, h / 2]]).reshape(-1, 1, 2)
    return float(np.abs(cv2.perspectiveTransform(c, H1) - cv2.perspectiveTransform(c, H2)).max())


class Registrar:
    """Tracks the (slowly drifting) camera pose over a video.

    Re-registers every `every` seconds; a failed fit keeps the last good
    homography, and a sudden jump larger than MAX_STEP px is accepted only
    when two consecutive fits agree on it (a real camera bump, not a bad fit).
    """
    MAX_STEP = 12.0

    def __init__(self, every: float = 5.0) -> None:
        self.every = every
        self.H = np.eye(3)
        self.have_fit = False
        self._next = 0.0
        self._pending: np.ndarray | None = None
        self._bank_hint = 0
        self.last_ok = False

    def update(self, t: float, frame_bgr: np.ndarray) -> bool:
        """Maybe re-register at time t; returns True if a fit was attempted."""
        if t < self._next:
            return False
        self._next = t + self.every
        H, _, ok = register(frame_bgr, prefer=self._bank_hint)
        self._bank_hint = register.last_bank
        self.last_ok = ok
        if not ok:
            return True
        if not self.have_fit or corner_shift(H, self.H) <= self.MAX_STEP:
            self.H, self.have_fit, self._pending = H, True, None
        elif self._pending is not None and corner_shift(H, self._pending) <= self.MAX_STEP:
            self.H, self._pending = H, None
        else:
            self._pending = H
        return True


def warp_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)


@dataclass
class Polygon:
    name: str
    pts: np.ndarray
    _mask: np.ndarray | None = field(default=None, repr=False)

    def mask(self) -> np.ndarray:
        if self._mask is None:
            w, h = config.ANALYSIS_SIZE
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [self.pts.astype(np.int32)], 1)
            self._mask = m.astype(bool)
        return self._mask

    def contains(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy).reshape(-1, 2)
        m = self.mask()
        x = np.clip(xy[:, 0].round().astype(int), 0, m.shape[1] - 1)
        y = np.clip(xy[:, 1].round().astype(int), 0, m.shape[0] - 1)
        inside = m[y, x]
        outside_frame = (xy[:, 0] < 0) | (xy[:, 1] < 0) | (xy[:, 0] >= m.shape[1]) | (xy[:, 1] >= m.shape[0])
        return inside & ~outside_frame


@lru_cache(maxsize=None)
def polygon(name: str) -> Polygon:
    layout = load_layout()
    return Polygon(name, np.array(layout["polygons"][name], dtype=np.float64))


def union_mask(names: list[str]) -> np.ndarray:
    m = np.zeros(config.ANALYSIS_SIZE[::-1], bool)
    for n in names:
        m |= polygon(n).mask()
    return m


def side_of_line(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed side of points p relative to the directed line a->b (>0 = left in image coords)."""
    p = np.asarray(p).reshape(-1, 2)
    return (b[0] - a[0]) * (p[:, 1] - a[1]) - (b[1] - a[1]) * (p[:, 0] - a[0])


def lane_angle(xy: np.ndarray) -> np.ndarray:
    """Angle (deg) of reference-frame ground points seen from the main road's vanishing point.

    Lane lines of the main road are rays from that point, so this is a
    perspective-free lateral (lane) coordinate on the approach and outbound.
    """
    vp = np.array(load_layout()["road_vp"]["xy"])
    d = np.asarray(xy, dtype=np.float64) - vp
    return np.degrees(np.arctan2(d[..., 1], d[..., 0]))
