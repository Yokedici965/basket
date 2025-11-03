"""
Base Event Detector Framework

Abstract base class for all event detectors.
New event types inherit from this and implement detect() method.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Event:
    """Basketball game event."""
    event_type: str
    confidence: float
    frame_start: int
    frame_end: int
    player_id: Optional[int] = None
    team: Optional[str] = None  # "home" or "away"
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type,
            "confidence": float(self.confidence),
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "player_id": self.player_id,
            "team": self.team,
            "metadata": self.metadata,
        }

    @property
    def duration_frames(self) -> int:
        """Duration in frames."""
        return self.frame_end - self.frame_start + 1


class BaseEventDetector(ABC):
    """
    Abstract base class for event detectors.

    All event types (Shot, Rebound, Steal, Block, etc.) inherit from this.

    Subclasses must implement:
    - detect(): Main detection logic
    """

    def __init__(self, event_type: str, fps: float = 30.0):
        """
        Initialize detector.

        Args:
            event_type: Name of event (e.g., "shot", "rebound")
            fps: Video frame rate
        """
        self.event_type = event_type
        self.fps = fps
        self.frame_time = 1.0 / fps

        # Configuration (can be overridden per subclass)
        self.min_confidence = 0.50
        self.min_duration_frames = 0
        self.max_duration_frames = 1000

    @abstractmethod
    def detect(self, frame_idx: int, detections: Dict,
              previous_events: List[Event]) -> Optional[Event]:
        """
        Detect event at given frame.

        Args:
            frame_idx: Current frame index
            detections: Detection data from YOLO/tracker
                {
                    "tracks": [(track_id, x, y, w, h, conf), ...],
                    "ball": (x, y, w, h, conf) or None,
                    "timestamp": float,
                }
            previous_events: List of previously detected events

        Returns:
            Event object if detected, None otherwise
        """
        pass

    def validate_event(self, event: Event) -> bool:
        """
        Validate event meets basic criteria.

        Can be overridden by subclasses for custom validation.

        Args:
            event: Event to validate

        Returns:
            True if event is valid
        """
        # Check confidence
        if event.confidence < self.min_confidence:
            return False

        # Check duration
        duration = event.duration_frames
        if duration < self.min_duration_frames:
            return False

        if duration > self.max_duration_frames:
            return False

        return True

    def get_last_event_of_type(self, previous_events: List[Event],
                              event_type: str) -> Optional[Event]:
        """
        Get most recent event of given type from history.

        Args:
            previous_events: List of previous events
            event_type: Event type to search for

        Returns:
            Most recent matching event, or None
        """
        for event in reversed(previous_events):
            if event.event_type == event_type:
                return event
        return None

    def frames_since_event(self, previous_events: List[Event],
                          event_type: str,
                          current_frame: int) -> Optional[int]:
        """
        Get number of frames since last event of given type.

        Args:
            previous_events: List of previous events
            event_type: Event type to search for
            current_frame: Current frame index

        Returns:
            Number of frames, or None if no such event
        """
        last_event = self.get_last_event_of_type(previous_events, event_type)
        if last_event is None:
            return None

        return current_frame - last_event.frame_end

    def frames_since_event_start(self, previous_events: List[Event],
                                 event_type: str,
                                 current_frame: int) -> Optional[int]:
        """Frames since start of last event of given type."""
        last_event = self.get_last_event_of_type(previous_events, event_type)
        if last_event is None:
            return None

        return current_frame - last_event.frame_start


class SimpleShotDetector(BaseEventDetector):
    """
    Simple shot detector using shot_classifier output.

    This is a placeholder - actual implementation would integrate
    with ShotClassifier module.
    """

    def __init__(self):
        super().__init__("shot_attempt")
        self.min_confidence = 0.70

    def detect(self, frame_idx: int, detections: Dict,
              previous_events: List[Event]) -> Optional[Event]:
        """
        Detect shot attempt.

        In real implementation, would:
        1. Extract ball trajectory from detections
        2. Call ShotClassifier.classify()
        3. Return Event if shot detected
        """
        # Placeholder: always return None
        # Real implementation integrates with shot_classifier.py
        return None


class SimplePossessionDetector(BaseEventDetector):
    """
    Simple possession detector.

    Detects when a team/player gains ball control.
    """

    def __init__(self):
        super().__init__("possession")
        self.min_confidence = 0.75
        self.min_duration_frames = 30  # At least 1 second

    def detect(self, frame_idx: int, detections: Dict,
              previous_events: List[Event]) -> Optional[Event]:
        """
        Detect possession change.

        Rules:
        1. Ball is detected
        2. Ball near a player
        3. Different from previous possession
        """
        ball = detections.get("ball")
        tracks = detections.get("tracks", [])

        if ball is None or len(tracks) == 0:
            return None

        # Simple heuristic: ball within 50 pixels of a player
        ball_x, ball_y = ball[0], ball[1]

        player_has_ball = False
        closest_track_id = None

        for track_id, x, y, w, h, conf in tracks:
            # Check if ball is near player bounding box
            player_center_x = x + w / 2
            player_center_y = y + h / 2

            distance = np.sqrt((ball_x - player_center_x) ** 2 +
                             (ball_y - player_center_y) ** 2)

            if distance < 50:  # Pixels
                player_has_ball = True
                closest_track_id = track_id
                break

        if not player_has_ball:
            return None

        # Check if this is a new possession (different from last)
        last_poss = self.get_last_event_of_type(previous_events, "possession")

        if last_poss and last_poss.player_id == closest_track_id:
            # Same player still has ball
            return None

        # New possession detected
        return Event(
            event_type="possession",
            confidence=0.80,
            frame_start=frame_idx,
            frame_end=frame_idx,
            player_id=closest_track_id,
            metadata={
                "ball_detected": True,
                "player_center": (ball_x, ball_y),
            }
        )


class SimpleReboundDetector(BaseEventDetector):
    """
    Simple rebound detector.

    Detects player securing ball after missed shot.
    """

    def __init__(self):
        super().__init__("rebound")
        self.min_confidence = 0.75

    def detect(self, frame_idx: int, detections: Dict,
              previous_events: List[Event]) -> Optional[Event]:
        """
        Detect rebound.

        Rules:
        1. Previous event was shot attempt (within 2 seconds)
        2. Current event is possession
        3. Different team from shooter
        """
        # Check if shot attempt happened recently
        frames_since_shot = self.frames_since_event_start(
            previous_events, "shot_attempt", frame_idx
        )

        if frames_since_shot is None or frames_since_shot > 60:  # 2 seconds
            return None

        # Check if possession is detected
        ball = detections.get("ball")
        tracks = detections.get("tracks", [])

        if ball is None or len(tracks) == 0:
            return None

        # Simplified: just mark as rebound if conditions met
        return Event(
            event_type="rebound",
            confidence=0.75,
            frame_start=frame_idx,
            frame_end=frame_idx,
            metadata={
                "frames_since_shot": frames_since_shot,
            }
        )


# Registry of event detectors
EVENT_DETECTOR_REGISTRY = {
    "possession": SimplePossessionDetector,
    "shot_attempt": SimpleShotDetector,
    "rebound": SimpleReboundDetector,
}


def get_detector(event_type: str) -> BaseEventDetector:
    """
    Get detector instance for given event type.

    Args:
        event_type: Event type name

    Returns:
        Instantiated detector

    Raises:
        ValueError if event type not registered
    """
    if event_type not in EVENT_DETECTOR_REGISTRY:
        raise ValueError(f"Unknown event type: {event_type}")

    detector_class = EVENT_DETECTOR_REGISTRY[event_type]
    return detector_class()


def register_detector(event_type: str, detector_class: type) -> None:
    """
    Register custom event detector.

    Args:
        event_type: Name of event type
        detector_class: Subclass of BaseEventDetector
    """
    if not issubclass(detector_class, BaseEventDetector):
        raise TypeError(f"{detector_class} must be subclass of BaseEventDetector")

    EVENT_DETECTOR_REGISTRY[event_type] = detector_class
