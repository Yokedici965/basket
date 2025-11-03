from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class Possession:
    start_frame: int
    end_frame: int
    start_ts: float
    end_ts: float
    controller_track: int
    controller_dist_mean: float
    frame_count: int


@dataclass
class ShotEvent:
    frame: int
    ts: float
    shooter_track: Optional[int]
    velocity: float
    result: str  # "made", "miss", "unknown"


@dataclass
class ReboundEvent:
    frame: int
    ts: float
    rebounder_track: Optional[int]
    related_shot_frame: Optional[int]


class EventEngine:
    BALL_CLS = 32
    PERSON_CLS = 0

    def __init__(
        self,
        *,
        ball_gap_sec: float = 1.0,
        min_possession_frames: int = 2,
        possession_dist_threshold: float = 140.0,
        hoops: Optional[List[Dict[str, float]]] = None,
        hoop_radius: float = 140.0,
        shot_speed_threshold: float = 55.0,
        shot_up_threshold: float = 25.0,
        rebound_window_sec: float = 2.5,
        made_window_sec: float = 1.2,
    ) -> None:
        self.ball_gap_sec = ball_gap_sec
        self.min_possession_frames = min_possession_frames
        self.possession_dist_threshold = possession_dist_threshold
        self.hoops = self._normalize_hoops(hoops)
        self.hoop_radius = hoop_radius
        self.shot_speed_threshold = shot_speed_threshold
        self.shot_up_threshold = shot_up_threshold
        self.rebound_window_sec = rebound_window_sec
        self.made_window_sec = made_window_sec

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_hoops(hoops: Optional[List[Dict[str, float]]]) -> List[Dict[str, float]]:
        result: List[Dict[str, float]] = []
        if not hoops:
            return result
        for idx, entry in enumerate(hoops):
            if not isinstance(entry, dict):
                continue
            try:
                x = float(entry["x"])
                y = float(entry["y"])
            except (KeyError, TypeError, ValueError):
                continue
            name = entry.get("name", f"hoop_{idx}")
            result.append({"name": name, "x": x, "y": y})
        return result

    @staticmethod
    def _distance(ax: float, ay: float, bx: float, by: float) -> float:
        return math.hypot(ax - bx, ay - by)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_tracks(self, tracks_csv: Path) -> pd.DataFrame:
        if not tracks_csv.exists():
            raise FileNotFoundError(tracks_csv)
        df = pd.read_csv(tracks_csv)
        expected_cols = {"frame", "ts", "cls", "track_id", "cx", "cy", "is_ref"}
        missing = expected_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {tracks_csv.name}: {sorted(missing)}")
        return df

    def load_detections(self, det_csv: Path) -> Optional[pd.DataFrame]:
        if not det_csv.exists():
            return None
        return pd.read_csv(det_csv)

    # ------------------------------------------------------------------
    # Possessions
    # ------------------------------------------------------------------
    def _split_tracks(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        ball_df = df[df["cls"] == self.BALL_CLS].copy()
        player_df = df[(df["cls"] == self.PERSON_CLS) & (df["is_ref"] == 0)].copy()
        ball_df = ball_df.sort_values("frame").reset_index(drop=True)
        player_df = player_df.sort_values(["frame", "track_id"]).reset_index(drop=True)
        return {"ball": ball_df, "players": player_df}

    def _nearest_players(self, ball_df: pd.DataFrame, player_df: pd.DataFrame) -> pd.DataFrame:
        if ball_df.empty:
            return pd.DataFrame(columns=["frame", "ts", "track_id", "dist"])
        merged = ball_df.merge(player_df, on="frame", suffixes=("_ball", "_player"))
        merged["dist"] = ((merged["cx_ball"] - merged["cx_player"]) ** 2 + (merged["cy_ball"] - merged["cy_player"]) ** 2) ** 0.5
        closest = (
            merged.sort_values("dist")
            .groupby("frame")
            .first()
            .reset_index()[["frame", "ts_ball", "track_id_player", "dist"]]
            .rename(columns={"ts_ball": "ts", "track_id_player": "track_id"})
        )
        return closest

    def compute_possessions(self, tracks_df: pd.DataFrame) -> tuple[List[Possession], pd.DataFrame, pd.DataFrame]:
        split = self._split_tracks(tracks_df)
        ball_df = split["ball"]
        player_df = split["players"]
        nearest = self._nearest_players(ball_df, player_df)
        possessions: List[Possession] = []
        if nearest.empty:
            return possessions, nearest, ball_df

        current: Optional[Dict[str, Any]] = None
        for row in nearest.itertuples(index=False):
            frame = int(row.frame)
            ts = float(row.ts)
            track_id = int(row.track_id)
            dist = float(row.dist)

            if current is None:
                current = {
                    "start_frame": frame,
                    "start_ts": ts,
                    "track_id": track_id,
                    "last_frame": frame,
                    "last_ts": ts,
                    "distances": [dist],
                }
                continue

            gap_ts = ts - current["last_ts"]
            id_change = track_id != current["track_id"]
            dist_break = dist > self.possession_dist_threshold

            if gap_ts > self.ball_gap_sec or id_change or dist_break:
                duration_frames = current["last_frame"] - current["start_frame"] + 1
                if duration_frames >= self.min_possession_frames:
                    possessions.append(
                        Possession(
                            start_frame=current["start_frame"],
                            end_frame=current["last_frame"],
                            start_ts=current["start_ts"],
                            end_ts=current["last_ts"],
                            controller_track=current["track_id"],
                            controller_dist_mean=sum(current["distances"]) / len(current["distances"]),
                            frame_count=duration_frames,
                        )
                    )
                current = {
                    "start_frame": frame,
                    "start_ts": ts,
                    "track_id": track_id,
                    "last_frame": frame,
                    "last_ts": ts,
                    "distances": [dist],
                }
            else:
                current["last_frame"] = frame
                current["last_ts"] = ts
                current["distances"].append(dist)

        if current is not None:
            duration_frames = current["last_frame"] - current["start_frame"] + 1
            if duration_frames >= self.min_possession_frames:
                possessions.append(
                    Possession(
                        start_frame=current["start_frame"],
                        end_frame=current["last_frame"],
                        start_ts=current["start_ts"],
                        end_ts=current["last_ts"],
                        controller_track=current["track_id"],
                        controller_dist_mean=sum(current["distances"]) / len(current["distances"]),
                        frame_count=duration_frames,
                    )
                )

        return possessions, nearest, ball_df

    # ------------------------------------------------------------------
    # Shot / rebound detection
    # ------------------------------------------------------------------
    def detect_shots_and_rebounds(
        self,
        ball_df: pd.DataFrame,
        nearest_players: pd.DataFrame,
    ) -> Dict[str, List[Any]]:
        shots: List[ShotEvent] = []
        rebounds: List[ReboundEvent] = []
        if ball_df.empty:
            return {"shots": shots, "rebounds": rebounds}

        work = ball_df.sort_values("frame").reset_index(drop=True).copy()
        work["dx"] = work["cx"].diff()
        work["dy"] = work["cy"].diff()
        work["speed"] = (work["dx"].fillna(0) ** 2 + work["dy"].fillna(0) ** 2) ** 0.5

        nearest_map = nearest_players.set_index("frame")[["track_id", "dist"]]
        used_frames: set[int] = set()

        for idx in range(1, len(work)):
            row = work.iloc[idx]
            frame = int(row["frame"])
            if frame in used_frames:
                continue

            dy = float(row["dy"])
            speed = float(row["speed"])
            if dy >= -self.shot_up_threshold or speed < self.shot_speed_threshold:
                continue

            ts = float(row["ts"])
            shooter_track = None
            if frame in nearest_map.index:
                shooter_track = int(nearest_map.loc[frame, "track_id"])

            shot_event = ShotEvent(
                frame=frame,
                ts=ts,
                shooter_track=shooter_track,
                velocity=speed,
                result="unknown",
            )
            shots.append(shot_event)

            made = False
            rebound_found = False
            for idx2 in range(idx + 1, len(work)):
                row2 = work.iloc[idx2]
                ts2 = float(row2["ts"])
                delta_t = ts2 - ts
                if delta_t > self.rebound_window_sec:
                    break

                frame2 = int(row2["frame"])

                if not made and delta_t <= self.made_window_sec and self.hoops:
                    for hoop in self.hoops:
                        if self._distance(row2["cx"], row2["cy"], hoop["x"], hoop["y"]) <= self.hoop_radius:
                            shot_event.result = "made"
                            made = True
                            break
                    if made:
                        used_frames.add(frame2)
                        break

                if frame2 in nearest_map.index:
                    rebounder_track = int(nearest_map.loc[frame2, "track_id"])
                    rebounds.append(
                        ReboundEvent(
                            frame=frame2,
                            ts=ts2,
                            rebounder_track=rebounder_track,
                            related_shot_frame=frame,
                        )
                    )
                    shot_event.result = "miss"
                    rebound_found = True
                    used_frames.add(frame2)
                    break

            used_frames.add(frame)
            if made:
                continue
            if rebound_found:
                continue

        return {"shots": shots, "rebounds": rebounds}

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export(
        self,
        *,
        possessions: List[Possession],
        shots: List[ShotEvent],
        rebounds: List[ReboundEvent],
        base_name: str,
        out_dir: Path,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        poss_records = [p.__dict__ for p in possessions]
        shots_records = [s.__dict__ for s in shots]
        rebounds_records = [r.__dict__ for r in rebounds]
        pd.DataFrame(poss_records).to_csv(out_dir / f"{base_name}_possessions.csv", index=False)
        pd.DataFrame(shots_records).to_csv(out_dir / f"{base_name}_shots.csv", index=False)
        pd.DataFrame(rebounds_records).to_csv(out_dir / f"{base_name}_rebounds.csv", index=False)
