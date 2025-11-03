"""
Event Sequencer

Manages event ordering, temporal validation, and conflict resolution.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from .base_detector import Event


@dataclass
class EventSequenceRule:
    """Rule for valid event sequences."""
    preceding_event: str
    following_event: str
    min_gap_frames: int = 0
    max_gap_frames: int = 1000000
    must_be_same_team: bool = False
    must_be_different_team: bool = False


class EventSequencer:
    """
    Manages basketball game event sequences.

    Responsibilities:
    1. Temporal ordering (events happen in time)
    2. Conflict resolution (overlapping events)
    3. Sequence validation (valid game flows)
    4. Event merging (merge adjacent similar events)
    """

    # Valid event sequences in basketball
    VALID_SEQUENCES = [
        # Shot sequences
        EventSequenceRule("possession", "shot_attempt", max_gap_frames=300),
        EventSequenceRule("shot_attempt", "shot_make", max_gap_frames=120),
        EventSequenceRule("shot_attempt", "shot_miss", max_gap_frames=120),
        EventSequenceRule("shot_attempt", "block", max_gap_frames=60),

        # Rebound sequences
        EventSequenceRule("shot_miss", "rebound", max_gap_frames=180),
        EventSequenceRule("rebound", "possession", max_gap_frames=60),

        # Turnover sequences
        EventSequenceRule("possession", "turnover", max_gap_frames=300),
        EventSequenceRule("turnover", "possession", max_gap_frames=60,
                         must_be_different_team=True),

        # Steal sequences
        EventSequenceRule("possession", "steal", max_gap_frames=300),
        EventSequenceRule("steal", "possession", max_gap_frames=60,
                         must_be_different_team=True),

        # Assist sequence
        EventSequenceRule("possession", "assist", max_gap_frames=300),
        EventSequenceRule("assist", "shot_make", max_gap_frames=120),
    ]

    def __init__(self):
        """Initialize sequencer."""
        self.events: List[Event] = []
        self.fps = 30.0
        self.validation_rules = self.VALID_SEQUENCES.copy()

    def add_event(self, event: Event) -> None:
        """
        Add event and resolve conflicts.

        Args:
            event: Event to add
        """
        # Handle overlaps with existing events
        self.events = self._resolve_overlaps(self.events, event)

        # Add to list (maintain temporal order)
        self.events.append(event)
        self.events.sort(key=lambda e: e.frame_start)

    def add_events(self, events: List[Event]) -> None:
        """Add multiple events."""
        for event in events:
            self.add_event(event)

    def _resolve_overlaps(self, existing_events: List[Event],
                         new_event: Event) -> List[Event]:
        """
        Remove/modify events that overlap with new event.

        Strategy:
        1. If new event has higher confidence, remove overlapping old events
        2. If old event has higher confidence, don't add new event
        3. If similar confidence, keep more specific event

        Args:
            existing_events: Current events
            new_event: Event to add

        Returns:
            Modified event list
        """
        result = []

        for existing in existing_events:
            # Check for temporal overlap
            if self._events_overlap(existing, new_event):
                # Same event type?
                if existing.event_type == new_event.event_type:
                    # Keep higher confidence
                    if existing.confidence > new_event.confidence:
                        result.append(existing)
                    else:
                        result.append(new_event)
                else:
                    # Different event types - both can coexist
                    # (e.g., shot_attempt and block during same time)
                    result.append(existing)
            else:
                result.append(existing)

        return result

    @staticmethod
    def _events_overlap(event1: Event, event2: Event) -> bool:
        """Check if two events overlap temporally."""
        return not (event1.frame_end < event2.frame_start or
                   event2.frame_end < event1.frame_start)

    def validate_sequence(self) -> bool:
        """
        Validate that event sequence follows basketball logic.

        Returns:
            True if valid, False otherwise
        """
        for i in range(len(self.events) - 1):
            current = self.events[i]
            next_event = self.events[i + 1]

            if not self._is_valid_transition(current, next_event):
                return False

        return True

    def _is_valid_transition(self, current: Event,
                            next_event: Event) -> bool:
        """
        Check if transition from current to next event is valid.

        Args:
            current: Current event
            next_event: Following event

        Returns:
            True if transition is valid
        """
        frame_gap = next_event.frame_start - current.frame_end

        # Check against validation rules
        for rule in self.validation_rules:
            if (rule.preceding_event == current.event_type and
                rule.following_event == next_event.event_type):

                # Check gap
                if not (rule.min_gap_frames <= frame_gap <= rule.max_gap_frames):
                    return False

                # Check team rules
                if rule.must_be_same_team:
                    if current.team != next_event.team:
                        return False

                if rule.must_be_different_team:
                    if current.team == next_event.team:
                        return False

                return True

        # If no specific rule, allow transition (assume valid)
        return True

    def get_events(self) -> List[Event]:
        """Get all events in chronological order."""
        return sorted(self.events, key=lambda e: e.frame_start)

    def get_events_by_type(self, event_type: str) -> List[Event]:
        """Get all events of specific type."""
        return [e for e in self.events if e.event_type == event_type]

    def get_events_in_range(self, start_frame: int,
                           end_frame: int) -> List[Event]:
        """Get events within frame range."""
        return [e for e in self.events
                if e.frame_start >= start_frame and e.frame_end <= end_frame]

    def merge_adjacent_events(self, event_type: str,
                             max_gap_frames: int = 30) -> None:
        """
        Merge adjacent events of same type if gap is small.

        Args:
            event_type: Event type to merge
            max_gap_frames: Max frames between events to merge
        """
        events_of_type = self.get_events_by_type(event_type)

        if len(events_of_type) < 2:
            return

        # Find gaps
        merged = []
        current_group = [events_of_type[0]]

        for i in range(1, len(events_of_type)):
            prev_event = events_of_type[i - 1]
            curr_event = events_of_type[i]

            gap = curr_event.frame_start - prev_event.frame_end

            if gap <= max_gap_frames:
                # Merge with current group
                current_group.append(curr_event)
            else:
                # Start new group
                merged.append(current_group)
                current_group = [curr_event]

        merged.append(current_group)

        # Replace in events list
        self.events = [e for e in self.events if e.event_type != event_type]

        for group in merged:
            merged_event = Event(
                event_type=event_type,
                confidence=max(e.confidence for e in group),
                frame_start=group[0].frame_start,
                frame_end=group[-1].frame_end,
                player_id=group[0].player_id,
                team=group[0].team,
                metadata={
                    "merged_count": len(group),
                    "original_confidences": [e.confidence for e in group],
                }
            )
            self.events.append(merged_event)

        self.events.sort(key=lambda e: e.frame_start)

    def get_sequence_string(self, max_events: int = 20) -> str:
        """
        Get human-readable sequence string.

        Args:
            max_events: Max events to show

        Returns:
            String representation
        """
        events = self.get_events()[:max_events]

        sequence = " → ".join([f"{e.event_type}({e.confidence:.2f})"
                               for e in events])

        if len(self.events) > max_events:
            sequence += f" ... (+{len(self.events) - max_events} more)"

        return sequence

    def to_dict(self) -> Dict:
        """Convert sequence to dictionary."""
        return {
            "total_events": len(self.events),
            "events": [e.to_dict() for e in self.get_events()],
            "event_types": list(set(e.event_type for e in self.events)),
            "sequence": self.get_sequence_string(),
        }
