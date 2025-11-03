"""
Camera Model and Calibration Module

Handles 2D pixel ↔ 3D court coordinate transformations using homography.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from pathlib import Path


class CameraModel:
    """
    Camera calibration and 2D-to-3D transformation using homography.

    The homography matrix H maps 2D pixel coordinates to 3D court coordinates
    on the court plane (Z=0).
    """

    # Court standard dimensions (in cm)
    COURT_3D_POINTS = {
        "rim_center": np.array([0, 0, 305]),  # 10 feet = 305 cm
        "backboard_base": np.array([0, -15.24, 0]),  # 6 inches from baseline
        "key_left": np.array([-244, 0, 0]),  # 16 feet / 2
        "key_right": np.array([244, 0, 0]),
        "three_point_arc": 725,  # 23.75 feet in cm
        "baseline": 0,
        "sideline": 457,  # 15 feet in cm
    }

    def __init__(self, video_path: Optional[str] = None):
        """
        Initialize camera model.

        Args:
            video_path: Path to video for extracting frame samples
        """
        self.video_path = video_path
        self.H = None  # Homography matrix (3x3)
        self.H_inv = None  # Inverse homography
        self.is_calibrated = False

    def calibrate_auto(self, frame: np.ndarray) -> bool:
        """
        Attempt automatic calibration by detecting court lines.

        Args:
            frame: Single video frame (BGR image)

        Returns:
            True if calibration successful
        """
        # This is a placeholder - actual implementation would use:
        # 1. Lane detection (Hough lines)
        # 2. Court line intersection detection
        # 3. SIFT/ORB keypoint matching with court model

        # For now, return False and fall back to manual calibration
        return False

    def calibrate_manual(self, pixel_points: np.ndarray,
                        court_points: np.ndarray) -> bool:
        """
        Manual calibration using correspondence point pairs.

        Args:
            pixel_points: (N, 2) array of pixel coordinates [x, y]
            court_points: (N, 2) array of court coordinates [X, Y]
                         (Z is assumed 0 for ground plane)

        Returns:
            True if calibration successful
        """
        if len(pixel_points) < 4:
            raise ValueError("Need at least 4 point correspondences")

        # Compute homography using DLT (Direct Linear Transform)
        self.H, mask = cv2.findHomography(pixel_points, court_points,
                                         method=cv2.RANSAC, ransacReprojThreshold=5.0)

        if self.H is None:
            return False

        self.H_inv = np.linalg.inv(self.H)
        self.is_calibrated = True

        # Validate calibration quality
        reprojection_error = self._compute_reprojection_error(pixel_points, court_points)
        if reprojection_error > 5.0:
            print(f"Warning: High reprojection error {reprojection_error:.2f} pixels")

        return True

    def _compute_reprojection_error(self, pixel_points: np.ndarray,
                                   court_points: np.ndarray) -> float:
        """
        Compute average reprojection error after homography.

        Args:
            pixel_points: (N, 2) pixel coordinates
            court_points: (N, 2) court coordinates

        Returns:
            Mean Euclidean error in pixels
        """
        reprojected = cv2.perspectiveTransform(pixel_points.reshape(-1, 1, 2), self.H)
        reprojected = reprojected.reshape(-1, 2)

        errors = np.linalg.norm(reprojected - court_points, axis=1)
        return float(np.mean(errors))

    def pixel_to_court(self, pixel_coords: np.ndarray) -> np.ndarray:
        """
        Transform 2D pixel coordinates to 3D court coordinates.

        Args:
            pixel_coords: (N, 2) array of [x, y] pixel positions
                         or single [x, y] coordinate

        Returns:
            court_coords: (N, 3) array of [X, Y, Z] court positions
                         or single [X, Y, Z] if input was 1D
        """
        if not self.is_calibrated:
            raise RuntimeError("Camera model not calibrated. Call calibrate_manual() first.")

        single_point = False
        if pixel_coords.ndim == 1:
            pixel_coords = pixel_coords.reshape(1, -1)
            single_point = True

        # Add homogeneous coordinate
        ones = np.ones((pixel_coords.shape[0], 1))
        pixel_coords_h = np.hstack([pixel_coords, ones])

        # Apply homography: court_coords = H @ pixel_coords_h
        court_2d = self.H @ pixel_coords_h.T  # (3, N)
        court_2d = court_2d[0:2] / court_2d[2]  # Normalize homogeneous coords
        court_2d = court_2d.T  # (N, 2)

        # Add Z=0 (ground plane)
        Z = np.zeros((court_2d.shape[0], 1))
        court_3d = np.hstack([court_2d, Z])

        if single_point:
            return court_3d[0]
        return court_3d

    def court_to_pixel(self, court_coords: np.ndarray) -> np.ndarray:
        """
        Transform 3D court coordinates to 2D pixel coordinates (inverse).

        Args:
            court_coords: (N, 3) array of [X, Y, Z] court positions
                         Assumes Z=0 for all points (ground plane)

        Returns:
            pixel_coords: (N, 2) array of [x, y] pixel positions
        """
        if not self.is_calibrated:
            raise RuntimeError("Camera model not calibrated.")

        single_point = False
        if court_coords.ndim == 1:
            court_coords = court_coords.reshape(1, -1)
            single_point = True

        # Use only X, Y (drop Z since we assume ground plane)
        court_2d = court_coords[:, :2]

        # Add homogeneous coordinate
        ones = np.ones((court_2d.shape[0], 1))
        court_2d_h = np.hstack([court_2d, ones])

        # Apply inverse homography
        pixel = self.H_inv @ court_2d_h.T
        pixel = pixel[0:2] / pixel[2]  # Normalize
        pixel = pixel.T  # (N, 2)

        if single_point:
            return pixel[0]
        return pixel

    def get_rim_position_3d(self) -> np.ndarray:
        """Get 3D court position of the rim (basket center)."""
        return self.COURT_3D_POINTS["rim_center"].copy()

    def get_rim_position_pixel(self) -> np.ndarray:
        """Get pixel position of the rim."""
        rim_3d = self.get_rim_position_3d()
        return self.court_to_pixel(rim_3d)

    def save(self, filepath: str) -> None:
        """Save calibration to file."""
        if self.H is None:
            raise RuntimeError("No calibration to save")
        np.save(filepath, self.H)

    def load(self, filepath: str) -> None:
        """Load calibration from file."""
        self.H = np.load(filepath)
        self.H_inv = np.linalg.inv(self.H)
        self.is_calibrated = True


def calibration_ui(video_path: str) -> CameraModel:
    """
    Interactive UI for manual camera calibration.

    User clicks on court positions in video frame, then specifies
    their 3D court coordinates.

    Returns:
        Calibrated CameraModel instance
    """
    # This is a placeholder for interactive calibration UI
    # Would use OpenCV mouse callback and cv2.imshow()
    raise NotImplementedError("Calibration UI to be implemented")
