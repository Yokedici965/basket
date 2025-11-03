from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import itertools
import math

import numpy as np
try:
    from filterpy.kalman import KalmanFilter
except ImportError:  # pragma: no cover
    KalmanFilter = None  # type: ignore


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def _create_kalman_filter(cx: float, cy: float, w: float, h: float) -> Optional[KalmanFilter]:
    if KalmanFilter is None:
        return None
    kf = KalmanFilter(dim_x=6, dim_z=4)
    dt = 1.0
    kf.F = np.array(
        [
            [1, 0, dt, 0, 0, 0],
            [0, 1, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=float,
    )
    kf.H = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=float,
    )
    kf.P *= 10.0
    kf.R = np.diag([15.0, 15.0, 30.0, 30.0])
    kf.Q *= 0.01
    kf.x = np.array([[cx], [cy], [0.0], [0.0], [w], [h]], dtype=float)
    return kf


def _state_to_bbox(x: np.ndarray) -> Tuple[float, float, float, float]:
    cx, cy, _, _, w, h = x.flatten()
    w = max(w, 1.0)
    h = max(h, 1.0)
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


@dataclass
class Track:
    tid: int
    cls: int
    bbox: Tuple[float, float, float, float]
    age: int = 0
    hit: int = 0
    conf: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    kf: Optional[KalmanFilter] = None


class IOUTracker:
    def __init__(self, iou_thr=0.35, max_age=15, center_dist_thr: float = 75.0):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.center_dist_thr = center_dist_thr
        self.tracks: Dict[int, Track] = {}
        self._next_id = itertools.count(1)

    def update(self, dets: List[Tuple[int, Tuple[float, float, float, float], float]]):
        """dets: list of (cls, bbox[x1,y1,x2,y2], conf)"""
        results: List[Dict[str, float]] = []

        for track in self.tracks.values():
            track.age += 1
            if track.kf is not None:
                track.kf.predict()
                track.bbox = _state_to_bbox(track.kf.x)
                track.cx = float(track.kf.x[0])
                track.cy = float(track.kf.x[1])

        matched_tracks = set()

        for cls, box, conf in dets:
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            best_iou, best_tid = 0.0, None
            best_center_dist = None
            for tid, tr in self.tracks.items():
                if tid in matched_tracks:
                    continue
                iou = iou_xyxy(box, tr.bbox)
                center_dist = math.hypot(cx - tr.cx, cy - tr.cy)
                if iou > best_iou:
                    best_iou, best_tid = iou, tid
                    best_center_dist = center_dist

            if best_tid is not None and (
                best_iou >= self.iou_thr or (best_center_dist is not None and best_center_dist <= self.center_dist_thr)
            ):
                tr = self.tracks[best_tid]
                if tr.kf is not None:
                    z = np.array([cx, cy, box[2] - box[0], box[3] - box[1]], dtype=float)
                    tr.kf.update(z)
                    tr.bbox = _state_to_bbox(tr.kf.x)
                    tr.cx = float(tr.kf.x[0])
                    tr.cy = float(tr.kf.x[1])
                else:
                    tr.bbox = box
                    tr.cx = cx
                    tr.cy = cy
                tr.cls = cls
                tr.conf = conf
                tr.age = 0
                tr.hit += 1
                matched_tracks.add(best_tid)
                results.append(
                    {
                        "cls": cls,
                        "bbox": tr.bbox,
                        "track_id": best_tid,
                        "conf": conf,
                        "age": tr.age,
                        "hits": tr.hit,
                        "cx": tr.cx,
                        "cy": tr.cy,
                        "matched": True,
                    }
                )
            else:
                new_id = next(self._next_id)
                w = box[2] - box[0]
                h = box[3] - box[1]
                kf = _create_kalman_filter(cx, cy, w, h)
                track = Track(
                    tid=new_id,
                    cls=cls,
                    bbox=box,
                    age=0,
                    hit=1,
                    conf=conf,
                    cx=cx,
                    cy=cy,
                    kf=kf,
                )
                self.tracks[new_id] = track
                matched_tracks.add(new_id)
                results.append(
                    {
                        "cls": cls,
                        "bbox": box,
                        "track_id": new_id,
                        "conf": conf,
                        "age": 0,
                        "hits": 1,
                        "cx": cx,
                        "cy": cy,
                        "matched": True,
                    }
                )

        for tid, tr in list(self.tracks.items()):
            if tid in matched_tracks:
                continue
            if tr.age > self.max_age:
                del self.tracks[tid]
                continue
            results.append(
                {
                    "cls": tr.cls,
                    "bbox": tr.bbox,
                    "track_id": tid,
                    "conf": tr.conf,
                    "age": tr.age,
                    "hits": tr.hit,
                    "cx": tr.cx,
                    "cy": tr.cy,
                    "matched": False,
                }
            )

        return results
