from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import yaml

BASE = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BASE / "configs" / "salon.yaml"


def load_qc(path: Path) -> List[Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    aggregated = payload.get("aggregated") or []
    if not aggregated:
        raise ValueError(f"No aggregated hoop entries found in {path}")
    return aggregated


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def update_config(cfg: Dict, hoops: List[Dict]) -> Dict:
    court = cfg.setdefault("court", {})
    current = court.setdefault("hoops", [])
    current_by_name = {str(entry.get("name")): entry for entry in current if "name" in entry}

    for entry in hoops:
        name = str(entry.get("name", "hoop"))
        updated = {
            "name": name,
            "x": round(float(entry["x"]), 2),
            "y": round(float(entry["y"]), 2),
        }
        if "radius" in entry:
            updated["radius"] = round(float(entry["radius"]), 2)
        current_by_name[name] = {**current_by_name.get(name, {}), **updated}

    court["hoops"] = list(current_by_name.values())
    return cfg


def write_config(path: Path, cfg: Dict) -> None:
    path.write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update salon.yaml hoops from calibration QC JSON.")
    parser.add_argument("--qc", required=True, type=Path, help="Path to *_calibration_qc.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Target YAML (default: configs/salon.yaml)")
    parser.add_argument("--dry-run", action="store_true", help="Print proposed updates without modifying YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hoops = load_qc(args.qc)
    cfg = load_config(args.config)
    updated = update_config(cfg, hoops)

    if args.dry_run:
        print(yaml.dump(updated.get("court", {}).get("hoops", []), sort_keys=False))
        return

    write_config(args.config, updated)
    print(f"[UPDATED] {args.config} hoops from {args.qc}")


if __name__ == "__main__":
    main()

