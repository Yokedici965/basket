# app/run.py
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2, numpy as np, pandas as pd, yaml
from ultralytics import YOLO

from app.utils.geo import point_in_poly
from app.utils.iou_tracker import IOUTracker

# Paths
BASE = Path(__file__).resolve().parents[1]
VIDEO_DIR = BASE / "input_videos"
OUT_DIR = BASE / "outputs"
CFG_PATH = BASE / "configs" / "salon.yaml"
MODELS_DIR = BASE / "models"

# Defaults
DEFAULT_CONF_BASE = 0.15
DEFAULT_IOU_NMS = 0.45
DEFAULT_IMG_SIZE = 832
DEFAULT_FRAME_STRIDE = 10
DEFAULT_LOG_EVERY_N = 600
DEFAULT_USE_GPU = True
DEFAULT_HALF = True
DEFAULT_PERSON_CLS = 0
DEFAULT_BALL_CLS = 32
DEFAULT_PERSON_THR = 0.25
DEFAULT_BALL_THR = 0.08
DEFAULT_STATIC_FRAMES = 15
DEFAULT_STATIC_GRID = 5
DEFAULT_MODEL_NAME = "yolov8n.pt"


def resolve_model_path(model_value: str) -> Path:
    candidate = Path(model_value)
    if candidate.is_absolute():
        return candidate
    direct = BASE / candidate
    if direct.exists():
        return direct
    models_dir = MODELS_DIR / candidate
    if models_dir.exists():
        return models_dir
    if candidate.exists():
        return candidate
    return MODELS_DIR / candidate.name


def _validate_polygon(poly: Sequence[Sequence[float]]) -> bool:
    return bool(poly) and all(len(p) == 2 for p in poly)


def validate_config(cfg: Dict) -> None:
    errors: List[str] = []

    court_cfg = cfg.get("court")
    if not isinstance(court_cfg, dict):
        errors.append("court section is required")
    else:
        polygon = court_cfg.get("polygon", [])
        if not _validate_polygon(polygon):
            errors.append("court.polygon must be a list of [x, y] pairs")

    detection_cfg = cfg.get("detection", {})
    classes_cfg = detection_cfg.get("classes", {})
    if not isinstance(classes_cfg, dict):
        errors.append("detection.classes must be a mapping with person/ball ids")
    else:
        for key in ("person", "ball"):
            if key not in classes_cfg:
                errors.append(f"detection.classes.{key} is required")
    static_cfg = detection_cfg.get("static_ball", {})
    if static_cfg:
        if int(static_cfg.get("frames", DEFAULT_STATIC_FRAMES)) < 1:
            errors.append("detection.static_ball.frames must be >= 1")
        if int(static_cfg.get("grid_px", DEFAULT_STATIC_GRID)) < 1:
            errors.append("detection.static_ball.grid_px must be >= 1")

    tracking_cfg = cfg.get("tracking", {})
    if tracking_cfg and float(tracking_cfg.get("iou_threshold", 0.0)) <= 0.0:
        errors.append("tracking.iou_threshold must be > 0")

    processing_cfg = cfg.get("processing", {})
    if int(processing_cfg.get("frame_stride", DEFAULT_FRAME_STRIDE)) < 1:
        errors.append("processing.frame_stride must be >= 1")

    if errors:
        bullet = "\n  - "
        raise ValueError(f"Config validation failed:{bullet}{bullet.join(errors)}")


def load_config() -> Tuple[Dict, Path]:
    if not CFG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CFG_PATH}")
    with open(CFG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    validate_config(cfg)
    model_value = cfg.get("model", DEFAULT_MODEL_NAME)
    model_path = resolve_model_path(model_value)
    return cfg, model_path


def get_device_kw(use_gpu: bool, half_precision: bool):
    try:
        import torch
        if use_gpu and torch.cuda.is_available():
            print(f"[INFO] CUDA: True | Device: {torch.cuda.get_device_name(0)} | FP16: {half_precision}")
            return dict(device=0, half=half_precision)
        print("[INFO] CUDA: False (running on CPU)")
    except Exception as exc:
        print(f"[WARN] CUDA check failed: {exc} (CPU will be used)")
    return {}


def process_video(path: Path, cfg: dict, model_path: Path, device_kw: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    court_cfg = cfg.get("court", {})
    court_poly = [tuple(p) for p in court_cfg.get("polygon", [])]
    if not court_poly:
        raise ValueError("Config field court.polygon is required")

    proc_cfg = cfg.get("processing", {})
    frame_stride = max(1, int(proc_cfg.get("frame_stride", DEFAULT_FRAME_STRIDE)))
    save_detections = bool(proc_cfg.get("save_detections", True))
    start_sec = max(0.0, float(proc_cfg.get("start_time_sec", 0.0)))
    duration_sec = proc_cfg.get("duration_sec")
    duration_sec = float(duration_sec) if duration_sec not in (None, "") else None
    if duration_sec is not None:
        duration_sec = max(0.0, duration_sec)
    log_every_n = max(1, int(proc_cfg.get("log_every_n", DEFAULT_LOG_EVERY_N)))

    detection_cfg = cfg.get("detection", {})
    conf_base = float(detection_cfg.get("base_conf", DEFAULT_CONF_BASE))
    nms_iou = float(detection_cfg.get("nms_iou", DEFAULT_IOU_NMS))
    img_size = int(detection_cfg.get("img_size", DEFAULT_IMG_SIZE))
    thresholds_cfg = detection_cfg.get("thresholds", {})
    person_thr = float(thresholds_cfg.get("person", DEFAULT_PERSON_THR))
    ball_thr = float(thresholds_cfg.get("ball", DEFAULT_BALL_THR))
    static_cfg = detection_cfg.get("static_ball", {})
    static_ball_enabled = bool(static_cfg.get("enabled", False))
    static_ball_frames = max(1, int(static_cfg.get("frames", DEFAULT_STATIC_FRAMES)))
    static_ball_grid = max(1, int(static_cfg.get("grid_px", DEFAULT_STATIC_GRID)))

    classes_cfg = detection_cfg.get("classes", {})
    person_cls = int(classes_cfg.get("person", DEFAULT_PERSON_CLS))
    ball_cls = int(classes_cfg.get("ball", DEFAULT_BALL_CLS))
    allowed = detection_cfg.get("allowed_classes")
    if allowed is None:
        allowed = {person_cls, ball_cls}
    else:
        allowed = {int(cls) for cls in allowed}
    tracking_cfg = cfg.get("tracking", {})
    tracker_iou = float(tracking_cfg.get("iou_threshold", 0.4))
    tracker_max_age = int(tracking_cfg.get("max_age", 25))
    tracker_center = float(tracking_cfg.get("center_distance_threshold", 75.0))

    print(f"[RUN] {path.name}")
    print(
        "[INFO] frame_stride=%s | save_detections=%s | log_every_n=%s"
        % (frame_stride, save_detections, log_every_n)
    )
    print(
        "[INFO] detection: conf_base=%.2f | iou=%.2f | img=%d | thr(person)=%.2f | thr(ball)=%.2f"
        % (conf_base, nms_iou, img_size, person_thr, ball_thr)
    )
    if static_ball_enabled:
        print(
            "[INFO] static_ball filter: frames>=%d | grid_px=%d"
            % (static_ball_frames, static_ball_grid)
        )
    print(
        "[INFO] tracking: iou=%.2f | max_age=%d | center_dist=%.1f"
        % (tracker_iou, tracker_max_age, tracker_center)
    )
    model = YOLO(str(model_path))
    tracker = IOUTracker(iou_thr=tracker_iou, max_age=tracker_max_age, center_dist_thr=tracker_center)
    static_ball_map: Dict[Tuple[int, int], Dict[str, int]] = {}
    static_prune_interval = static_ball_frames * 10
    static_ball_flagged = 0

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print("[ERR] Video could not be opened:", path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    start_frame = int(start_sec * fps)
    if start_frame >= total_frames:
        print(f"[WARN] start_frame {start_frame} total frames {total_frames}")
        return
    end_frame = None
    if duration_sec is not None:
        duration_frames = int(duration_sec * fps)
        end_frame = min(total_frames, start_frame + duration_frames)
    frames_to_process = total_frames - start_frame if end_frame is None else max(0, end_frame - start_frame)
    expected_steps = (frames_to_process + frame_stride - 1) // frame_stride if frames_to_process > 0 else 0
    print(f"[INFO] FPS: {fps} | Frames: {total_frames} | start={start_frame} | limit={'full' if end_frame is None else end_frame} | expected steps: {expected_steps}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if frames_to_process <= 0:
        print("[WARN] Configured window produced no frames")
        return

    track_rows = []
    det_rows = []
    processed_steps = 0
    f = start_frame
    t0 = time.time()
    last_log_f = -log_every_n

    while True:
        if end_frame is not None and f >= end_frame:
            break
        ok, frame = cap.read()
        if not ok:
            break

        if f % frame_stride == 0:
            pred = model.predict(
                frame,
                imgsz=img_size,
                conf=conf_base,
                iou=nms_iou,
                verbose=False,
                **device_kw,
            )[0]
            ts = f / fps
            processed_steps += 1

            dets = []
            for box in pred.boxes:
                cls = int(box.cls.item())
                if cls not in allowed:
                    continue

                conf = float(box.conf.item())
                if cls == person_cls and conf < person_thr:
                    continue
                if cls == ball_cls and conf < ball_thr:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if cls == person_cls and not point_in_poly((cx, cy), court_poly):
                    continue

                is_static_ball = 0
                if static_ball_enabled and cls == ball_cls:
                    key = (int(cx // static_ball_grid), int(cy // static_ball_grid))
                    entry = static_ball_map.get(key)
                    if entry and f - entry["last_frame"] <= static_ball_frames:
                        entry["count"] += 1
                    else:
                        entry = {"count": 1}
                    entry["last_frame"] = f
                    static_ball_map[key] = entry
                    if entry["count"] >= static_ball_frames:
                        static_ball_flagged += 1
                        is_static_ball = 1

                dets.append((cls, (x1, y1, x2, y2), conf))
                det_rows.append(
                    {
                        "frame": f,
                        "ts": round(ts, 3),
                        "cls": cls,
                        "conf": round(conf, 4),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "cx": cx,
                        "cy": cy,
                        "is_static": is_static_ball,
                    }
                )

            track_results = tracker.update(dets)
            for tr in track_results:
                if not tr.get("matched", True) and tr["cls"] != ball_cls:
                    continue
                x1, y1, x2, y2 = tr["bbox"]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = (x2 - x1), (y2 - y1)
                is_static_track = 0
                if static_ball_enabled and tr["cls"] == ball_cls:
                    key = (int(cx // static_ball_grid), int(cy // static_ball_grid))
                    info = static_ball_map.get(key)
                    if info and f - info["last_frame"] <= static_ball_frames and info["count"] >= static_ball_frames:
                        is_static_track = 1
                is_predicted = 0 if tr.get("matched", True) else 1
                track_rows.append(
                    {
                        "frame": f,
                        "ts": round(ts, 3),
                        "cls": tr["cls"],
                        "track_id": tr["track_id"],
                        "conf": round(tr["conf"], 4),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "cx": cx,
                        "cy": cy,
                        "w": w,
                        "h": h,
                        "age": tr["age"],
                        "hits": tr["hits"],
                        "is_static": is_static_track,
                        "is_predicted": is_predicted,
                        "is_ref": 0,
                    }
                )

            if static_ball_enabled and static_ball_map and f % max(static_prune_interval, 1) == 0:
                obsolete = [key for key, info in static_ball_map.items() if f - info["last_frame"] > static_ball_frames]
                for key in obsolete:
                    static_ball_map.pop(key, None)

        if f - last_log_f >= log_every_n:
            elapsed = time.time() - t0
            speed = processed_steps / elapsed if elapsed > 0 else 0.0
            remain = max(expected_steps - processed_steps, 0)
            eta_min = (remain / speed / 60) if speed > 0 else 0.0
            print(
                f"[PROGRESS] frame {f}/{total_frames} | processed {processed_steps} | "
                f"speed {speed:.2f} it/s | ETA ~{eta_min:.1f} min"
            )
            last_log_f = f

        f += 1

    cap.release()

    track_df = pd.DataFrame(track_rows)
    out_tracks = OUT_DIR / f"{path.name}_tracks.csv"
    if not track_df.empty:
        track_df.to_csv(out_tracks, index=False)
        print(f"[OK] {len(track_rows)} track rows -> {out_tracks} (elapsed {time.time() - t0:.1f}s)")
    else:
        print(f"[WARN] No track rows were produced (elapsed {time.time() - t0:.1f}s)")

    if save_detections:
        det_df = pd.DataFrame(det_rows)
        out_dets = OUT_DIR / f"{path.name}_detections.csv"
        if not det_df.empty:
            det_df.to_csv(out_dets, index=False)
            print(f"[INFO] {len(det_rows)} detection rows -> {out_dets}")
        else:
            print("[WARN] No detection rows were produced")
    if static_ball_enabled:
        print(f"[INFO] static_ball flagged detections: {static_ball_flagged}")
    return {"static_ball_flagged": static_ball_flagged}


def main():
    print("[INFO] BASE:", BASE)
    print("[INFO] VIDEO_DIR:", VIDEO_DIR, "exists?", VIDEO_DIR.exists())
    print("[INFO] OUT_DIR:", OUT_DIR, "exists?", OUT_DIR.exists())
    print("[INFO] CFG:", CFG_PATH, "exists?", CFG_PATH.exists())

    cfg, model_path = load_config()
    print("[INFO] MODEL:", model_path, "exists?", model_path.exists())

    vids = sorted(VIDEO_DIR.glob("*.mp4"))
    if not vids:
        print("[HINT] Drop .mp4 files into input_videos (example: game1.mp4)")
        return

    detection_cfg = cfg.get("detection", {})
    use_gpu = bool(detection_cfg.get("use_gpu", DEFAULT_USE_GPU))
    half_precision = bool(detection_cfg.get("half_precision", DEFAULT_HALF))
    device_kw = get_device_kw(use_gpu=use_gpu, half_precision=half_precision)
    for video in vids:
        process_video(video, cfg, model_path, device_kw)


if __name__ == "__main__":
    main()

