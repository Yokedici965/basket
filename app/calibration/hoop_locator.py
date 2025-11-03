from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

try:  # Ultralytics is optional for fallback model inference
    from ultralytics import YOLO
except Exception:  # pragma: no cover - if ultralytics not installed, skip fallback
    YOLO = None


@dataclass
class FrameSample:
    frame_idx: int
    ts: float
    frame: np.ndarray


@dataclass
class HoopCandidate:
    name: str
    x: float
    y: float
    radius: float
    confidence: float
    method: str
    frame_idx: int
    ts: float


@dataclass
class HoopSummary:
    name: str
    x: float
    y: float
    radius: float
    confidence: float
    methods: List[str]
    sample_count: int
    std_x: float
    std_y: float
    std_radius: float
    ts_span: float


HSV_LOWER = np.array([5, 80, 80])
HSV_UPPER = np.array([25, 255, 255])


def _sample_frames(cap: cv2.VideoCapture, step_frames: int, max_samples: int) -> List[FrameSample]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    samples: List[FrameSample] = []
    frame_idx = 0
    while frame_idx < total_frames and len(samples) < max_samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        ts = frame_idx / fps
        samples.append(FrameSample(frame_idx=frame_idx, ts=ts, frame=frame))
        frame_idx += step_frames
    return samples


def _compute_homographies(reference: np.ndarray, frames: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float]]:
    orb = cv2.ORB_create(1000)
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
    if des_ref is None or len(kp_ref) < 20:
        identity = np.eye(3, dtype=np.float32)
        return [identity for _ in frames], [0.0 for _ in frames]
    homographies: List[np.ndarray] = []
    confidences: List[float] = []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        if des is None or len(kp) < 20:
            homographies.append(np.eye(3, dtype=np.float32))
            confidences.append(0.0)
            continue
        matches = bf.match(des_ref, des)
        matches = sorted(matches, key=lambda m: m.distance)[:80]
        if len(matches) < 10:
            homographies.append(np.eye(3, dtype=np.float32))
            confidences.append(0.0)
            continue
        src = np.float32([kp_ref[m.queryIdx].pt for m in matches])
        dst = np.float32([kp[m.trainIdx].pt for m in matches])
        H, mask = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
        if H is None:
            homographies.append(np.eye(3, dtype=np.float32))
            confidences.append(0.0)
        else:
            homographies.append(H.astype(np.float32))
            if mask is None:
                confidences.append(0.0)
            else:
                inliers = float(np.count_nonzero(mask))
                confidences.append(inliers / max(len(mask), 1))
    return homographies, confidences


def _warp(frame: np.ndarray, H: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.warpPerspective(frame, H, size, flags=cv2.INTER_LINEAR)


def _detect_circles(frame: np.ndarray, frame_idx: int, ts: float, width: int) -> List[HoopCandidate]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200, param1=120, param2=15, minRadius=40, maxRadius=160)
    results: List[HoopCandidate] = []
    if circles is not None:
        for x, y, r in circles[0]:
            region = mask[int(max(y - r, 0)): int(min(y + r, mask.shape[0])), int(max(x - r, 0)): int(min(x + r, mask.shape[1]))]
            fill = float(np.count_nonzero(region)) / max(region.size, 1)
            side = "left" if x <= width / 2 else "right"
            results.append(
                HoopCandidate(
                    name=side,
                    x=float(x),
                    y=float(y),
                    radius=float(r),
                    confidence=float(np.clip(fill, 0.0, 1.0)),
                    method="hough",
                    frame_idx=frame_idx,
                    ts=ts,
                )
            )
    return results


def _detect_with_yolo(
    frame: np.ndarray,
    *,
    frame_idx: int,
    ts: float,
    width: int,
    model: Optional[Any],
    conf: float,
) -> List[HoopCandidate]:
    if model is None:
        return []
    try:
        predictions = model.predict(frame, conf=conf, verbose=False)
    except Exception:
        return []
    if not predictions:
        return []
    result = predictions[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    cands: List[HoopCandidate] = []
    for box in boxes:
        cls_name = None
        if hasattr(result, "names"):
            names = result.names or {}
            cls_idx = int(getattr(box, "cls", [0])[0])
            cls_name = names.get(cls_idx, str(cls_idx))
        if cls_name:
            label = cls_name.lower()
            if not any(token in label for token in ("hoop", "basket", "rim")):
                continue
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        conf_val = float(getattr(box, "conf", [0.0])[0])
        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        radius = max((x2 - x1), (y2 - y1)) / 2.0
        side = "left" if cx <= width / 2 else "right"
        cands.append(
            HoopCandidate(
                name=side,
                x=float(cx),
                y=float(cy),
                radius=float(radius),
                confidence=float(np.clip(conf_val, 0.0, 1.0)),
                method="yolo",
                frame_idx=frame_idx,
                ts=ts,
            )
        )
    return cands


def _smooth_series(values: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    if values.size == 0:
        return values
    smoothed = np.empty_like(values)
    running = values[0]
    smoothed[0] = running
    for idx in range(1, values.size):
        running = alpha * values[idx] + (1.0 - alpha) * running
        smoothed[idx] = running
    return smoothed


def _fuse_candidates(candidates: List[HoopCandidate], width: int) -> Tuple[List[HoopSummary], Dict[str, Any]]:
    if not candidates:
        return [], {"counts": {}, "groups": {}}

    groups: Dict[str, List[HoopCandidate]] = defaultdict(list)
    for cand in candidates:
        side = cand.name or ("left" if cand.x <= width / 2 else "right")
        groups[side].append(cand)

    summaries: List[HoopSummary] = []
    group_stats: Dict[str, Any] = {}
    for side, group in groups.items():
        ordered = sorted(group, key=lambda c: (c.ts, c.frame_idx))
        xs = np.array([c.x for c in ordered], dtype=np.float32)
        ys = np.array([c.y for c in ordered], dtype=np.float32)
        rs = np.array([c.radius for c in ordered], dtype=np.float32)
        xs = _smooth_series(xs)
        ys = _smooth_series(ys)
        rs = _smooth_series(rs, alpha=0.5)
        ws = np.array([max(c.confidence, 1e-3) for c in group], dtype=np.float32)
        confs = np.array([c.confidence for c in group], dtype=np.float32)
        ts_vals = np.array([c.ts for c in ordered], dtype=np.float32)
        methods = sorted({c.method for c in group})

        x_med = float(np.median(xs))
        y_med = float(np.median(ys))
        r_med = float(np.median(rs))
        confidence = float(np.clip(np.average(confs, weights=ws), 0.0, 1.0))
        std_x = float(np.std(xs))
        std_y = float(np.std(ys))
        std_r = float(np.std(rs))
        ts_span = float(ts_vals.max() - ts_vals.min()) if len(ts_vals) > 1 else 0.0

        summaries.append(
            HoopSummary(
                name=side,
                x=x_med,
                y=y_med,
                radius=r_med,
                confidence=confidence,
                methods=methods,
                sample_count=len(group),
                std_x=std_x,
                std_y=std_y,
                std_radius=std_r,
                ts_span=ts_span,
            )
        )
        group_stats[side] = {
            "count": len(group),
            "methods": methods,
            "std_x": std_x,
            "std_y": std_y,
            "std_radius": std_r,
            "confidence_avg": confidence,
            "ts_span": ts_span,
        }

    counts = Counter(c.method for c in candidates)
    summaries.sort(key=lambda item: item.name)
    return summaries, {"counts": dict(counts), "groups": group_stats}


def calibrate_hoops(
    video_path: Path,
    *,
    config_path: Path,
    output_dir: Path,
    sample_stride_sec: float = 5.0,
    max_samples: int = 20,
    yolo_weights: Optional[Path] = None,
    yolo_conf: float = 0.25,
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video açılamadı: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride_frames = int(sample_stride_sec * fps)
    samples = _sample_frames(cap, stride_frames, max_samples)
    if not samples:
        raise RuntimeError("Video'dan örnek kare alınamadı")
    ref_frame = samples[0].frame
    homographies, homography_conf = _compute_homographies(ref_frame, [s.frame for s in samples])
    h, w = ref_frame.shape[:2]
    yolo_model = None
    if yolo_weights is not None:
        if YOLO is None:
            raise ImportError("Ultralytics yüklü değil; YOLO fallback kullanılamıyor")
        if not yolo_weights.exists():
            raise FileNotFoundError(f"YOLO model bulunamadı: {yolo_weights}")
        yolo_model = YOLO(str(yolo_weights))
    detections: List[HoopCandidate] = []
    low_homography_frames: List[int] = []
    for sample, H, hom_conf in zip(samples, homographies, homography_conf):
        warp = _warp(sample.frame, H, (w, h))
        detections.extend(_detect_circles(warp, sample.frame_idx, sample.ts, w))
        detections.extend(_detect_with_yolo(warp, frame_idx=sample.frame_idx, ts=sample.ts, width=w, model=yolo_model, conf=yolo_conf))
        if hom_conf < 0.1:
            low_homography_frames.append(sample.frame_idx)
    cap.release()

    summaries, diagnostics = _fuse_candidates(detections, w)
    qc = {
        "samples": len(samples),
        "raw_detections": len(detections),
        "homography_confidence": homography_conf,
        "low_homography_frames": low_homography_frames,
        "detection_stats": diagnostics,
        "aggregated": [s.__dict__ for s in summaries],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    qc_path = output_dir / f"{video_path.name}_calibration_qc.json"
    with qc_path.open("w", encoding="utf-8") as fh:
        json.dump(qc, fh, ensure_ascii=False, indent=2)

    if not summaries:
        return qc

    # update config
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    court = config.setdefault("court", {})
    court["hoops"] = [
        {"name": summary.name, "x": round(float(summary.x), 2), "y": round(float(summary.y), 2)}
        for summary in summaries
    ]
    events_cfg = config.setdefault("events", {})
    mean_radius = sum(summary.radius for summary in summaries) / len(summaries)
    events_cfg.setdefault("hoop_radius", round(float(mean_radius * 1.2), 2))

    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, allow_unicode=True, sort_keys=False)

    return qc
