"""
Feature Extraction Module for Shot Classification

Extracts 18-dimensional feature vector from ball trajectory.
"""

import numpy as np
from typing import Dict


class FeatureExtractor:
    """
    Extract ML-ready features from analyzed ball trajectory.

    18-dimensional feature vector:
    1. Initial velocity (m/s)
    2. Final velocity (m/s)
    3. Max velocity (m/s)
    4. Initial angle (degrees)
    5. Entry angle vertical (degrees)
    6. Entry angle horizontal (degrees)
    7. Apex height above rim (cm)
    8. Apex height (absolute, cm)
    9. Rim distance at apex (cm)
    10. Rim distance at start (cm)
    11. Rim distance at end (cm)
    12. Rim distance final / initial (ratio)
    13. Trajectory duration (seconds)
    14. Trajectory total distance (cm)
    15. Parabola fit quality (R²)
    16. Vertical velocity at impact (cm/s)
    17. Angle of descent (degrees)
    18. Distance traveled in Z (cm)
    """

    def __init__(self):
        """Initialize feature extractor."""
        self.feature_names = [
            "initial_velocity",
            "final_velocity",
            "max_velocity",
            "initial_angle_elevation",
            "entry_angle_vertical",
            "entry_angle_horizontal",
            "apex_height_above_rim",
            "apex_height_absolute",
            "rim_distance_at_apex",
            "rim_distance_at_start",
            "rim_distance_at_end",
            "rim_distance_ratio",
            "duration",
            "total_distance",
            "parabola_r_squared",
            "final_vertical_velocity",
            "angle_of_descent",
            "z_distance_traveled",
        ]

    def extract(self, trajectory_analysis: Dict,
               rim_position: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from trajectory analysis.

        Args:
            trajectory_analysis: Dict from TrajectoryAnalyzer.analyze()
            rim_position: (3,) 3D position of rim center

        Returns:
            (18,) feature vector
        """
        trajectory_3d = trajectory_analysis["trajectory_3d"]
        velocities = trajectory_analysis["velocities"]
        apex_info = trajectory_analysis["apex"]
        duration = trajectory_analysis["duration"]
        total_distance = trajectory_analysis["distance"]

        features = []

        # 1. Initial velocity (m/s)
        initial_velocity = float(np.linalg.norm(velocities[0])) / 100.0
        features.append(initial_velocity)

        # 2. Final velocity (m/s)
        final_velocity = float(np.linalg.norm(velocities[-1])) / 100.0
        features.append(final_velocity)

        # 3. Max velocity (m/s)
        speeds = np.linalg.norm(velocities, axis=1)
        max_velocity = float(np.max(speeds)) / 100.0
        features.append(max_velocity)

        # 4. Initial angle elevation (degrees)
        initial_angle_elev = self._compute_elevation_angle(velocities[0])
        features.append(initial_angle_elev)

        # 5, 6. Entry angles (vertical and horizontal)
        entry_angle_v, entry_angle_h = self._compute_entry_angles(velocities[-1])
        features.append(entry_angle_v)
        features.append(entry_angle_h)

        # 7. Apex height above rim
        RIM_HEIGHT = 305  # cm
        apex_height_above_rim = apex_info["apex_height"] - RIM_HEIGHT
        features.append(apex_height_above_rim)

        # 8. Apex height (absolute)
        features.append(apex_info["apex_height"])

        # 9-11. Rim distances at different points
        rim_dist_at_apex = np.linalg.norm(
            apex_info["apex_position"] - rim_position
        )
        rim_dist_at_start = np.linalg.norm(trajectory_3d[0] - rim_position)
        rim_dist_at_end = np.linalg.norm(trajectory_3d[-1] - rim_position)

        features.append(rim_dist_at_apex)
        features.append(rim_dist_at_start)
        features.append(rim_dist_at_end)

        # 12. Rim distance ratio (end / start)
        rim_dist_ratio = (
            rim_dist_at_end / rim_dist_at_start
            if rim_dist_at_start > 0
            else 1.0
        )
        features.append(rim_dist_ratio)

        # 13. Duration
        features.append(duration)

        # 14. Total distance
        features.append(total_distance / 100.0)  # Convert to meters

        # 15. Parabola fit quality
        # Placeholder - would compute from trajectory fitting
        parabola_r2 = 0.85  # Default decent fit
        features.append(parabola_r2)

        # 16. Final vertical velocity (cm/s)
        final_vertical_velocity = float(velocities[-1, 2])
        features.append(final_vertical_velocity)

        # 17. Angle of descent (degrees)
        angle_of_descent = self._compute_descent_angle(velocities[-1])
        features.append(angle_of_descent)

        # 18. Z distance traveled (cm)
        z_distance = abs(trajectory_3d[-1, 2] - trajectory_3d[0, 2])
        features.append(z_distance)

        return np.array(features, dtype=np.float32)

    def _compute_elevation_angle(self, velocity_vector: np.ndarray) -> float:
        """
        Compute elevation angle of velocity vector.

        Args:
            velocity_vector: (3,) [vx, vy, vz]

        Returns:
            Angle in degrees (0-90)
        """
        horizontal_speed = np.sqrt(velocity_vector[0] ** 2 + velocity_vector[1] ** 2)
        vertical_speed = velocity_vector[2]

        if horizontal_speed == 0:
            return 90.0 if vertical_speed > 0 else -90.0

        angle_rad = np.arctan2(vertical_speed, horizontal_speed)
        angle_deg = np.degrees(angle_rad)
        return float(angle_deg)

    def _compute_entry_angles(self, velocity_vector: np.ndarray) -> tuple:
        """
        Compute entry angles (vertical and horizontal).

        Args:
            velocity_vector: (3,) [vx, vy, vz]

        Returns:
            (vertical_angle, horizontal_angle) in degrees
        """
        # Vertical angle: angle from horizontal plane
        horizontal_speed = np.sqrt(velocity_vector[0] ** 2 + velocity_vector[1] ** 2)
        vertical_angle = np.degrees(np.arctan2(velocity_vector[2], horizontal_speed))

        # Horizontal angle: azimuth angle
        horizontal_angle = np.degrees(np.arctan2(velocity_vector[1], velocity_vector[0]))

        return float(vertical_angle), float(horizontal_angle)

    def _compute_descent_angle(self, velocity_vector: np.ndarray) -> float:
        """
        Compute angle of descent (angle from vertical).

        Args:
            velocity_vector: (3,) [vx, vy, vz]

        Returns:
            Angle in degrees (0-90)
        """
        horizontal_speed = np.sqrt(velocity_vector[0] ** 2 + velocity_vector[1] ** 2)
        vertical_speed = abs(velocity_vector[2])  # Use absolute for descent

        if vertical_speed == 0:
            return 0.0

        angle_rad = np.arctan2(horizontal_speed, vertical_speed)
        angle_deg = np.degrees(angle_rad)
        return float(angle_deg)

    def get_feature_names(self) -> list:
        """Return list of feature names."""
        return self.feature_names.copy()

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using standard scaling.

        In production, would use fitted StandardScaler from training.
        For now, uses approximate ranges.

        Args:
            features: (N, 18) or (18,) feature array

        Returns:
            Normalized features with same shape
        """
        # Approximate feature ranges (empirically determined)
        feature_stats = {
            "initial_velocity": (3.0, 2.0),  # (mean, std)
            "final_velocity": (0.5, 0.8),
            "max_velocity": (4.0, 1.5),
            "initial_angle_elevation": (45.0, 20.0),
            "entry_angle_vertical": (-30.0, 20.0),
            "entry_angle_horizontal": (0.0, 90.0),
            "apex_height_above_rim": (100.0, 80.0),
            "apex_height_absolute": (400.0, 100.0),
            "rim_distance_at_apex": (150.0, 100.0),
            "rim_distance_at_start": (300.0, 200.0),
            "rim_distance_at_end": (30.0, 50.0),
            "rim_distance_ratio": (0.2, 0.3),
            "duration": (1.5, 0.5),
            "total_distance": (3.0, 1.5),
            "parabola_r_squared": (0.85, 0.1),
            "final_vertical_velocity": (-200.0, 150.0),
            "angle_of_descent": (60.0, 20.0),
            "z_distance_traveled": (150.0, 100.0),
        }

        # Simple z-score normalization
        is_1d = features.ndim == 1
        if is_1d:
            features = features.reshape(1, -1)

        normalized = features.copy().astype(np.float32)

        for i, name in enumerate(self.feature_names):
            mean, std = feature_stats.get(name, (0.0, 1.0))
            if std > 0:
                normalized[:, i] = (features[:, i] - mean) / std

        if is_1d:
            normalized = normalized[0]

        return normalized
