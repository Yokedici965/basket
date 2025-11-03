from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _format_summary(agg: Dict[str, Any]) -> str:
    methods = ", ".join(agg.get("methods", []))
    return (
        f"{agg['name']:<6} "
        f"x={agg['x']:.1f} y={agg['y']:.1f} r={agg['radius']:.1f} "
        f"conf={agg['confidence']:.2f} "
        f"n={agg['sample_count']} std(x)={agg['std_x']:.2f} std(y)={agg['std_y']:.2f} "
        f"std(r)={agg['std_radius']:.2f} span={agg['ts_span']:.1f}s "
        f"methods=[{methods}]"
    )


def _detection_rate(samples: int, aggregated: Iterable[Dict[str, Any]]) -> float:
    total_hits = sum(item.get("sample_count", 0) for item in aggregated)
    if samples <= 0:
        return 0.0
    return min(1.0, total_hits / float(samples))


def _average_confidence(aggregated: Iterable[Dict[str, Any]]) -> float:
    confs = [float(item.get("confidence", 0.0)) for item in aggregated]
    if not confs:
        return 0.0
    return sum(confs) / len(confs)


def _fallback_usage(det_stats: Dict[str, Any]) -> Dict[str, int]:
    counts = det_stats.get("counts", {}) if det_stats else {}
    return {str(k): int(v) for k, v in counts.items()}


def _flag_summaries(
    aggregated: Iterable[Dict[str, Any]],
    *,
    conf_threshold: float = 0.5,
    std_threshold: float = 50.0,
) -> List[Dict[str, Any]]:
    flags: List[Dict[str, Any]] = []
    for summary in aggregated:
        reasons: List[str] = []
        conf = float(summary.get("confidence", 0.0))
        std_x = float(summary.get("std_x", 0.0))
        std_y = float(summary.get("std_y", 0.0))
        if conf < conf_threshold:
            reasons.append(f"low confidence {conf:.2f}")
        if std_x > std_threshold or std_y > std_threshold:
            reasons.append(f"coordinate jitter std_x={std_x:.1f} std_y={std_y:.1f}")
        if reasons:
            flags.append(
                {
                    "type": "aggregate",
                    "name": summary.get("name", "unknown"),
                    "frame_idx": "",
                    "ts": "",
                    "reason": "; ".join(reasons),
                }
            )
    return flags


def _flag_missing_hoops(aggregated: Iterable[Dict[str, Any]], expected: int = 2) -> Optional[Dict[str, Any]]:
    agg_list = list(aggregated)
    if len(agg_list) >= expected:
        return None
    return {
        "type": "aggregate",
        "name": "all",
        "frame_idx": "",
        "ts": "",
        "reason": f"expected {expected} hoops, found {len(agg_list)}",
    }


def _flag_low_homography(frames: Iterable[int]) -> List[Dict[str, Any]]:
    return [
        {
            "type": "homography",
            "name": "",
            "frame_idx": frame,
            "ts": "",
            "reason": "low homography confidence",
        }
        for frame in frames
    ]


def _export_flags(flags: List[Dict[str, Any]], export_path: Optional[Path]) -> None:
    if not export_path or not flags:
        return
    export_path.parent.mkdir(parents=True, exist_ok=True)
    header = "type,name,frame_idx,ts,reason\n"
    lines = [header]
    for flag in flags:
        line = ",".join(
            [
                str(flag.get("type", "")),
                str(flag.get("name", "")),
                str(flag.get("frame_idx", "")),
                str(flag.get("ts", "")),
                flag.get("reason", "").replace(",", ";"),
            ]
        )
        lines.append(line + "\n")
    export_path.write_text("".join(lines), encoding="utf-8")


def review_calibration(qc_path: Path, *, export_csv: Optional[Path] = None) -> Dict[str, Any]:
    if not qc_path.exists():
        raise FileNotFoundError(qc_path)
    data = json.loads(qc_path.read_text(encoding="utf-8"))
    aggregated: List[Dict[str, Any]] = data.get("aggregated", [])
    samples = int(data.get("samples", 0))
    raw_detections = int(data.get("raw_detections", 0))
    detection_rate = _detection_rate(samples, aggregated)
    avg_conf = _average_confidence(aggregated)
    fallback_counts = _fallback_usage(data.get("detection_stats", {}))
    print(f"[QC] samples={samples} raw_detections={raw_detections}")
    print(f"[QC] detection rate={detection_rate:.2%} avg_conf={avg_conf:.2f}")
    det_stats = data.get("detection_stats", {})
    counts = det_stats.get("counts", {})
    if counts:
        printable = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        print(f"[QC] detections per method -> {printable}")
    else:
        print("[QC] detections per method -> none recorded")
    low_frames = data.get("low_homography_frames", [])
    if low_frames:
        print(f"[QC] homography low confidence frames: {len(low_frames)} (min idx: {min(low_frames)}, max idx: {max(low_frames)})")
    else:
        print("[QC] homography low confidence frames: none")
    if not aggregated:
        print("[QC] No hoop summaries available.")
        flags: List[Dict[str, Any]] = _flag_low_homography(low_frames)
        _export_flags(flags, export_csv)
        return data
    print("[QC] Hoop summaries")
    for agg in aggregated:
        print("  -", _format_summary(agg))
    flags = _flag_summaries(aggregated)
    missing_flag = _flag_missing_hoops(aggregated)
    if missing_flag:
        flags.append(missing_flag)
    flags.extend(_flag_low_homography(low_frames))
    if flags:
        print(f"[QC] flagged entries -> {len(flags)} (export to CSV: {export_csv})")
    else:
        print("[QC] flagged entries -> none")
    _export_flags(flags, export_csv)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Review hoop calibration QC output.")
    parser.add_argument("--qc", required=True, help="Path to *_calibration_qc.json")
    parser.add_argument("--export-csv", help="Optional path to export flagged entries as CSV")
    args = parser.parse_args()
    qc_path = Path(args.qc)
    export = Path(args.export_csv) if args.export_csv else None
    review_calibration(qc_path, export_csv=export)


if __name__ == "__main__":
    main()
