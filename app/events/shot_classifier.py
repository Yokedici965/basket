"""
Shot Classifier - Heuristic and ML-based classification

Classifies ball trajectories into: MAKE, MISS, or ATTEMPT
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from .camera_model import CameraModel
from .trajectory_analyzer import TrajectoryAnalyzer
from .feature_extractor import FeatureExtractor


@dataclass
class ShotClassification:
    """Result of shot classification."""
    shot_type: str  # "make", "miss", "attempt"
    confidence: float  # 0.0-1.0
    features: np.ndarray  # 18-dimensional feature vector
    trajectory_id: str
    frame_start: int
    frame_end: int

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "shot_type": self.shot_type,
            "confidence": float(self.confidence),
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "trajectory_id": self.trajectory_id,
        }


class ShotClassifier:
    """
    Classifies shots as MAKE, MISS, or ATTEMPT.

    Uses two-phase approach:
    - Phase 1 (Week 2): Heuristic rules (65-70% accuracy)
    - Phase 2 (Week 3): ML model (80-85% accuracy)
    """

    # Class labels
    CLASS_LABELS = {0: "attempt", 1: "miss", 2: "make"}
    LABEL_TO_IDX = {"attempt": 0, "miss": 1, "make": 2}

    # Confidence thresholds
    MIN_CONFIDENCE = {
        "make": 0.70,
        "miss": 0.65,
        "attempt": 0.50,
    }

    def __init__(self, camera_model: CameraModel,
                ml_model_path: Optional[str] = None):
        """
        Initialize shot classifier.

        Args:
            camera_model: Calibrated CameraModel instance
            ml_model_path: Path to trained ML model (optional)
        """
        self.camera_model = camera_model
        self.trajectory_analyzer = TrajectoryAnalyzer()
        self.feature_extractor = FeatureExtractor()

        self.ml_model = None
        self.use_ml = False

        if ml_model_path and Path(ml_model_path).exists():
            self.load_ml_model(ml_model_path)

        # Rim position
        self.rim_position = camera_model.get_rim_position_3d()

    def classify(self, trajectory_3d: np.ndarray,
                frame_start: int,
                frame_end: int,
                trajectory_id: str = "unknown") -> ShotClassification:
        """
        Classify a shot trajectory.

        Args:
            trajectory_3d: (N, 3) ball trajectory in 3D court coordinates
            frame_start: Starting frame index
            frame_end: Ending frame index
            trajectory_id: Unique identifier for trajectory

        Returns:
            ShotClassification result
        """
        # Analyze trajectory
        traj_analysis = self.trajectory_analyzer.analyze(trajectory_3d)

        # Extract features
        features = self.feature_extractor.extract(
            traj_analysis, self.rim_position
        )

        # Classify
        if self.use_ml and self.ml_model is not None:
            shot_type, confidence = self._classify_ml(features)
        else:
            shot_type, confidence = self._classify_heuristic(
                traj_analysis, features
            )

        # Apply confidence threshold
        if confidence < self.MIN_CONFIDENCE.get(shot_type, 0.5):
            shot_type = "attempt"
            confidence = max(confidence, self.MIN_CONFIDENCE["attempt"])

        return ShotClassification(
            shot_type=shot_type,
            confidence=confidence,
            features=features,
            trajectory_id=trajectory_id,
            frame_start=frame_start,
            frame_end=frame_end,
        )

    def _classify_heuristic(self, traj_analysis: Dict,
                           features: np.ndarray) -> Tuple[str, float]:
        """
        Heuristic-based classification (Phase 1).

        Decision Rules:
        1. MAKE: Ball near rim at end + good trajectory
        2. MISS: Ball near rim at apex but far at end
        3. ATTEMPT: Doesn't meet make/miss criteria

        Returns:
            (shot_type, confidence)
        """
        confidence = 0.5  # Base confidence

        # Extract key features for rules
        rim_dist_at_apex = features[8]
        rim_dist_at_end = features[10]
        rim_dist_ratio = features[11]
        duration = features[12]
        apex_height_above_rim = features[6]
        entry_angle_vertical = features[4]
        final_vertical_velocity = features[15]

        # Rule 1: MAKE shot
        # - Very close to rim at end (<25 cm)
        # - Good entry angle (30-60 degrees down)
        # - Ball falling (negative vertical velocity)
        if (rim_dist_at_end < 25 and
            -60 < entry_angle_vertical < -30 and
            final_vertical_velocity < -100):
            confidence = 0.80
            return ("make", confidence)

        # Rule 2: MISS shot
        # - Close to rim at apex (<100 cm)
        # - But far from rim at end (>50 cm)
        # - Clear parabolic trajectory
        if (rim_dist_at_apex < 100 and
            rim_dist_at_end > 50 and
            apex_height_above_rim > 50 and
            duration > 0.8):
            confidence = 0.75
            return ("miss", confidence)

        # Rule 3: ATTEMPT (fallback)
        return ("attempt", 0.40)

    def _classify_ml(self, features: np.ndarray) -> Tuple[str, float]:
        """
        ML-based classification (Phase 2).

        Uses trained logistic regression model.

        Returns:
            (shot_type, confidence)
        """
        if self.ml_model is None:
            return ("attempt", 0.4)

        # Normalize features
        features_norm = self.feature_extractor.normalize_features(features)

        # Predict
        try:
            probs = self.ml_model.predict_proba([features_norm])[0]
            pred_idx = np.argmax(probs)
            confidence = float(probs[pred_idx])

            shot_type = self.CLASS_LABELS[pred_idx]
            return (shot_type, confidence)
        except Exception as e:
            print(f"ML prediction error: {e}")
            return ("attempt", 0.4)

    def load_ml_model(self, model_path: str) -> bool:
        """
        Load trained ML model (sklearn pipeline).

        Args:
            model_path: Path to pickled sklearn pipeline

        Returns:
            True if successful
        """
        try:
            import pickle

            with open(model_path, "rb") as f:
                self.ml_model = pickle.load(f)

            self.use_ml = True
            return True
        except Exception as e:
            print(f"Failed to load ML model: {e}")
            return False

    def save_ml_model(self, model_path: str) -> bool:
        """
        Save trained ML model.

        Args:
            model_path: Path to save pickled pipeline

        Returns:
            True if successful
        """
        if self.ml_model is None:
            return False

        try:
            import pickle

            with open(model_path, "wb") as f:
                pickle.dump(self.ml_model, f)

            return True
        except Exception as e:
            print(f"Failed to save ML model: {e}")
            return False

    @staticmethod
    def train_ml_model(X_train: np.ndarray, y_train: np.ndarray):
        """
        Train ML classifier from features and labels.

        Args:
            X_train: (N, 18) training features
            y_train: (N,) labels (0=attempt, 1=miss, 2=make)

        Returns:
            Trained sklearn Pipeline
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=500,
                class_weight="balanced",
                random_state=42,
            )),
        ])

        pipeline.fit(X_train, y_train)
        return pipeline

    def batch_classify(self, trajectories: list) -> list:
        """
        Classify multiple trajectories.

        Args:
            trajectories: List of dicts with:
                - "trajectory_3d": (N, 3) positions
                - "frame_start": int
                - "frame_end": int
                - "trajectory_id": str (optional)

        Returns:
            List of ShotClassification objects
        """
        results = []

        for traj in trajectories:
            result = self.classify(
                traj["trajectory_3d"],
                traj["frame_start"],
                traj["frame_end"],
                traj.get("trajectory_id", "unknown"),
            )
            results.append(result)

        return results
