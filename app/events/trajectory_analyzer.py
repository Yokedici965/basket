"""
Ball Trajectory Analyzer Module

Extracts kinematic properties from ball motion sequence.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from scipy.interpolate import UnivariateSpline


class TrajectoryAnalyzer:
    """
    Analyzes ball trajectory to extract kinematic features.

    Features computed:
    - Velocity (instantaneous and average)
    - Acceleration
    - Apex (highest point)
    - Arc fitting (parabolic)
    - Duration and distance
    """

    def __init__(self, fps: float = 30.0):
        """
        Initialize analyzer.

        Args:
            fps: Video frame rate (frames per second)
        """
        self.fps = fps
        self.frame_time = 1.0 / fps  # Time per frame in seconds

    def analyze(self, trajectory_3d: np.ndarray,
               frame_indices: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze complete ball trajectory.

        Args:
            trajectory_3d: (N, 3) array of [X, Y, Z] 3D positions
            frame_indices: (N,) frame indices (auto-generated if None)

        Returns:
            Dict with kinematic features:
                - velocities: (N-1, 3) velocity vectors
                - speeds: (N-1,) scalar speeds
                - accelerations: (N-2, 3) acceleration vectors
                - apex: dict with apex properties
                - duration: total trajectory time
                - distance: total distance traveled
                - validation: dict with validation checks
        """
        if len(trajectory_3d) < 3:
            raise ValueError("Need at least 3 points for trajectory analysis")

        if frame_indices is None:
            frame_indices = np.arange(len(trajectory_3d))

        # Compute timestamps
        timestamps = frame_indices / self.fps

        # Compute kinematics
        velocities = self._compute_velocity(trajectory_3d, timestamps)
        speeds = np.linalg.norm(velocities, axis=1)
        accelerations = self._compute_acceleration(velocities, timestamps)

        # Find apex
        apex_info = self.detect_apex(trajectory_3d)

        # Total duration and distance
        duration = timestamps[-1] - timestamps[0]
        distance = np.sum(np.linalg.norm(np.diff(trajectory_3d, axis=0), axis=1))

        # Validate trajectory
        validation = self.validate_trajectory(trajectory_3d, velocities, apex_info)

        return {
            "trajectory_3d": trajectory_3d,
            "timestamps": timestamps,
            "velocities": velocities,
            "speeds": speeds,
            "accelerations": accelerations,
            "apex": apex_info,
            "duration": duration,
            "distance": distance,
            "validation": validation,
        }

    def _compute_velocity(self, positions: np.ndarray,
                         timestamps: np.ndarray) -> np.ndarray:
        """
        Compute instantaneous velocity using central differences.

        Args:
            positions: (N, 3) trajectory
            timestamps: (N,) timestamps in seconds

        Returns:
            velocities: (N-1, 3) velocity vectors in cm/s
        """
        # Central difference: v(i) = (p(i+1) - p(i-1)) / (2*dt)
        # For edges, use forward/backward difference

        dt = np.diff(timestamps)  # (N-1,)
        dp = np.diff(positions, axis=0)  # (N-1, 3)

        # Velocity = dp / dt
        velocities = dp / dt[:, np.newaxis]  # (N-1, 3)

        return velocities

    def _compute_acceleration(self, velocities: np.ndarray,
                             timestamps: np.ndarray) -> np.ndarray:
        """
        Compute acceleration.

        Args:
            velocities: (N-1, 3) velocity vectors
            timestamps: (N,) original timestamps

        Returns:
            accelerations: (N-2, 3) acceleration vectors
        """
        dv = np.diff(velocities, axis=0)  # (N-2, 3)
        dt = np.diff(timestamps)[:-1]  # (N-2,)

        accelerations = dv / dt[:, np.newaxis]

        return accelerations

    def detect_apex(self, trajectory_3d: np.ndarray) -> Dict:
        """
        Detect the highest point (apex) of the trajectory.

        Returns:
            Dict with:
                - apex_idx: Frame index of apex
                - apex_height: Z-coordinate at apex (cm)
                - apex_position: [X, Y, Z] at apex
                - has_apex: True if valid apex found
                - apex_time: Time to reach apex (seconds)
        """
        Z = trajectory_3d[:, 2]  # Height component
        max_idx = np.argmax(Z)
        max_height = Z[max_idx]

        # Validation: apex should be > rim height + margin
        RIM_HEIGHT = 305  # cm
        MIN_APEX_HEIGHT = RIM_HEIGHT + 30  # At least 30cm above rim

        has_apex = max_height >= MIN_APEX_HEIGHT

        return {
            "apex_idx": int(max_idx),
            "apex_height": float(max_height),
            "apex_position": trajectory_3d[max_idx].astype(float),
            "has_apex": bool(has_apex),
            "apex_time": float(max_idx) / self.fps,
        }

    def validate_trajectory(self, trajectory_3d: np.ndarray,
                          velocities: np.ndarray,
                          apex_info: Dict) -> Dict:
        """
        Check if trajectory is physically plausible for a basketball shot.

        Validation Criteria:
        1. Must have valid apex
        2. Initial velocity reasonable (>200 cm/s, <1000 cm/s)
        3. Gravity effect visible (acceleration ~ -980 cm/s²)
        4. Duration reasonable (0.5-4 seconds)

        Returns:
            Dict with validation status for each criterion
        """
        checks = {}

        # Check 1: Valid apex
        checks["has_apex"] = apex_info["has_apex"]

        # Check 2: Initial velocity
        initial_speed = np.linalg.norm(velocities[0]) if len(velocities) > 0 else 0
        checks["initial_speed_valid"] = 200 <= initial_speed <= 1000
        checks["initial_speed"] = float(initial_speed)

        # Check 3: Gravity effect
        if len(velocities) > 1:
            # Average acceleration in Z direction should be ~ -980 cm/s²
            z_accelerations = np.diff(velocities[:, 2])
            mean_z_accel = float(np.mean(z_accelerations))
            checks["gravity_effect_valid"] = -1100 <= mean_z_accel <= -800
            checks["mean_z_acceleration"] = mean_z_accel
        else:
            checks["gravity_effect_valid"] = False
            checks["mean_z_acceleration"] = 0.0

        # Check 4: Duration
        duration = (len(trajectory_3d) - 1) / self.fps
        checks["duration_valid"] = 0.5 <= duration <= 4.0
        checks["duration"] = float(duration)

        # Overall validity: at least 3/4 checks pass
        num_passed = sum(1 for k, v in checks.items()
                        if k.endswith("_valid") and v)
        checks["is_valid"] = num_passed >= 3

        return checks

    def fit_parabola(self, trajectory_3d: np.ndarray,
                    frame_indices: np.ndarray) -> Dict:
        """
        Fit parabolic arc to trajectory.

        Assumes 2D parabola in XZ plane (ignores Y).

        Returns:
            Dict with parabola parameters (a, b, c):
                Z = a*X² + b*X + c
        """
        X = trajectory_3d[:, 0]
        Z = trajectory_3d[:, 2]

        # Fit polynomial (degree 2)
        try:
            coeffs = np.polyfit(X, Z, 2)
            a, b, c = coeffs

            # R² score
            y_pred = np.polyval(coeffs, X)
            ss_res = np.sum((Z - y_pred) ** 2)
            ss_tot = np.sum((Z - np.mean(Z)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return {
                "a": float(a),
                "b": float(b),
                "c": float(c),
                "r_squared": float(r_squared),
                "is_valid": r_squared > 0.8,
            }
        except np.linalg.LinAlgError:
            return {
                "a": 0.0,
                "b": 0.0,
                "c": 0.0,
                "r_squared": 0.0,
                "is_valid": False,
            }

    def get_rim_distance_at_frame(self, trajectory_3d: np.ndarray,
                                 rim_position: np.ndarray,
                                 frame_idx: int) -> float:
        """
        Compute 3D Euclidean distance from rim at specific frame.

        Args:
            trajectory_3d: (N, 3) ball positions
            rim_position: (3,) rim center position
            frame_idx: Frame index to query

        Returns:
            Distance in cm
        """
        if frame_idx < 0 or frame_idx >= len(trajectory_3d):
            return float('inf')

        ball_pos = trajectory_3d[frame_idx]
        distance = np.linalg.norm(ball_pos - rim_position)
        return float(distance)

    def get_closest_approach_to_rim(self, trajectory_3d: np.ndarray,
                                   rim_position: np.ndarray) -> Dict:
        """
        Find closest approach to rim during entire trajectory.

        Returns:
            Dict with:
                - min_distance: Minimum distance to rim (cm)
                - min_frame: Frame index of closest approach
                - position: [X, Y, Z] at closest approach
        """
        distances = np.linalg.norm(trajectory_3d - rim_position, axis=1)
        min_idx = np.argmin(distances)

        return {
            "min_distance": float(distances[min_idx]),
            "min_frame": int(min_idx),
            "position": trajectory_3d[min_idx].astype(float),
        }

    def extract_shot_window(self, trajectory_3d: np.ndarray,
                           apex_info: Dict,
                           window_before: int = 10,
                           window_after: int = 10) -> np.ndarray:
        """
        Extract trajectory window around apex (for shot detection).

        Args:
            trajectory_3d: Full trajectory
            apex_info: Apex information dict
            window_before: Frames before apex to include
            window_after: Frames after apex to include

        Returns:
            (M, 3) sub-trajectory around apex
        """
        apex_idx = apex_info["apex_idx"]
        start = max(0, apex_idx - window_before)
        end = min(len(trajectory_3d), apex_idx + window_after + 1)

        return trajectory_3d[start:end]
