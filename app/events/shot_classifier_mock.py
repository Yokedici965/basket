"""Mock shot classification pipeline for documentation and integration planning.

This module illustrates how calibrated hoop metadata could be consumed by the
real event engine once the production classifier is implemented.  It
intentionally avoids heavy dependencies (e.g. OpenCV, Torch) so it can run in
restricted environments and unit tests.

Example
-------
>>> from pathlib import Path
>>> from app.events.shot_classifier_mock import HoopModel, ShotClassifierMock
>>> hoop = HoopModel.from_calibration(Path('configs/calibrations/mac2.mp4_calibration_qc.json'))
>>> classifier = ShotClassifierMock(hoop)
>>> result = classifier.classify_shot([(1000, (1800.0, 900.0)), (1008, (1850.0, 640.0))])
>>> result['label']
'3PT'

The mock returns coarse labels based on planar distance to the nearest hoop
and the vertical displacement of the ball trajectory.  Replace this module with
`shot_classifier.py` once the full implementation is ready.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import json

HoopPoint = Tuple[float, float]
Trajectory = Sequence[Tuple[int, HoopPoint]]  # (frame_idx, (cx, cy)) pairs


@dataclass
class HoopModel:
    """Lightweight representation of calibrated hoops.

    Attributes
    ----------
    name:
        Identifier (``"left"`` or ``"right"``).
    position:
        2D coordinates in image space (pixels).
    radius:
        Average detected hoop radius (pixels).
    confidence:
        Aggregated confidence score from calibration.
    """

    name: str
    position: HoopPoint
    radius: float
    confidence: float

    @staticmethod
    def _from_summary(summary: Dict[str, float]) -> "HoopModel":
        return HoopModel(
            name=str(summary.get("name", "unknown")),
            position=(float(summary.get("x", 0.0)), float(summary.get("y", 0.0))),
            radius=float(summary.get("radius", 0.0)),
            confidence=float(summary.get("confidence", 0.0)),
        )

    @classmethod
    def from_calibration(cls, qc_path: Path) -> List["HoopModel"]:
        """Load hoop metadata from a calibration QC JSON file.

        Parameters
        ----------
        qc_path:
            Path to ``*_calibration_qc.json`` created by ``hoops_cli``.

        Returns
        -------
        list of :class:`HoopModel`
            One entry per detected hoop.  Returns an empty list if the file
            cannot be read or contains no aggregated data.
        """

        if not qc_path.exists():
            return []
        try:
            payload = json.loads(qc_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        aggregated = payload.get("aggregated") or []
        return [cls._from_summary(entry) for entry in aggregated]


@dataclass
class ShotInference:
    """Output schema used by :class:`ShotClassifierMock`."""

    label: str
    distance_px: float
    angle_deg: float
    hoop: Optional[str]

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "label": self.label,
            "distance_px": self.distance_px,
            "angle_deg": self.angle_deg,
            "hoop": self.hoop,
        }


class ShotClassifierMock:
    """Toy classifier that derives shot metadata from trajectories.

    Parameters
    ----------
    hoops:
        Iterable of :class:`HoopModel` instances sourced from calibration.
    three_pt_radius:
        Threshold in pixels separating 2PT vs 3PT attempts.  Default is 700 px
        which approximately matches a 3PT arc in 4K footage after homography
        normalisation.
    free_throw_band:
        Vertical band (y coordinate range) considered a free-throw.  When the
        ball starts inside this band and remains within it, the shot is flagged
        as ``"FT"``.
    """

    def __init__(
        self,
        hoops: Iterable[HoopModel],
        *,
        three_pt_radius: float = 700.0,
        free_throw_band: Tuple[float, float] = (800.0, 1200.0),
    ) -> None:
        self.hoops: List[HoopModel] = list(hoops)
        self.three_pt_radius = three_pt_radius
        self.free_throw_band = free_throw_band

    def _nearest_hoop(self, point: HoopPoint) -> Optional[HoopModel]:
        if not self.hoops:
            return None
        return min(self.hoops, key=lambda hoop: hypot(point[0] - hoop.position[0], point[1] - hoop.position[1]))

    def classify_shot(self, trajectory: Trajectory) -> ShotInference:
        """Assign a coarse-grained label to the provided trajectory.

        Parameters
        ----------
        trajectory:
            Ordered sequence of ``(frame_idx, (cx, cy))`` pairs representing the
            ball track for a single shot attempt.

        Returns
        -------
        :class:`ShotInference`
            Dataclass with the predicted label, Euclidean distance to the
            closest hoop, and launch angle in degrees.  Defaults to ``"unknown"``
            when no metadata is available.
        """

        if not trajectory:
            return ShotInference(label="unknown", distance_px=0.0, angle_deg=0.0, hoop=None)

        # Use the first observation as the release point.
        _, (start_x, start_y) = trajectory[0]
        hoop = self._nearest_hoop((start_x, start_y))
        if hoop is None:
            return ShotInference(label="unknown", distance_px=0.0, angle_deg=0.0, hoop=None)

        distance = hypot(start_x - hoop.position[0], start_y - hoop.position[1])
        # Approximate launch angle using final observation if present.
        end_x, end_y = trajectory[-1][1]
        angle = degrees(atan2(hoop.position[1] - end_y, hoop.position[0] - end_x))

        label = "2PT"
        if self.free_throw_band[0] <= start_y <= self.free_throw_band[1]:
            label = "FT"
        elif distance >= self.three_pt_radius:
            label = "3PT"

        return ShotInference(label=label, distance_px=distance, angle_deg=angle, hoop=hoop.name)


__all__ = [
    "HoopModel",
    "ShotInference",
    "ShotClassifierMock",
]
