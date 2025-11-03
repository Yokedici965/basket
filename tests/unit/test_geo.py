"""Unit tests for app/utils/geo.py geometry utilities."""
from __future__ import annotations

import numpy as np
import pytest
from app.utils.geo import point_in_poly, polyline


class TestPointInPoly:
    """Tests for point_in_poly function."""

    @pytest.mark.unit
    def test_point_inside_square(self):
        """Test point clearly inside square."""
        poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert point_in_poly((5, 5), poly) is True

    @pytest.mark.unit
    def test_point_outside_square(self):
        """Test point clearly outside square."""
        poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert point_in_poly((15, 15), poly) is False

    @pytest.mark.unit
    def test_point_on_edge(self):
        """Test point exactly on polygon edge."""
        poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert point_in_poly((5, 0), poly) is True

    @pytest.mark.unit
    def test_point_on_vertex(self):
        """Test point exactly on polygon vertex."""
        poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert point_in_poly((0, 0), poly) is True

    @pytest.mark.unit
    def test_complex_polygon(self):
        """Test with complex (concave) polygon."""
        poly = [(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)]
        assert point_in_poly((7, 2), poly) is True
        assert point_in_poly((7, 7), poly) is False

    @pytest.mark.unit
    def test_triangle(self):
        """Test with triangle polygon."""
        poly = [(0, 0), (10, 0), (5, 10)]
        assert point_in_poly((5, 3), poly) is True
        assert point_in_poly((1, 9), poly) is False

    @pytest.mark.unit
    def test_float_coordinates(self):
        """Test with floating point coordinates."""
        poly = [(0.5, 0.5), (10.5, 0.5), (10.5, 10.5), (0.5, 10.5)]
        assert point_in_poly((5.5, 5.5), poly) is True

    @pytest.mark.unit
    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        poly = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
        assert point_in_poly((0, 0), poly) is True
        assert point_in_poly((-20, 0), poly) is False


class TestPolyline:
    """Tests for polyline function."""

    @pytest.mark.unit
    def test_simple_polyline(self):
        """Test creating polyline from simple points."""
        points = [(0, 0), (10, 0), (10, 10)]
        result = polyline(points)

        assert result.shape == (3, 1, 2)
        assert result.dtype == np.int32
        np.testing.assert_array_equal(result[:, 0, :], [[0, 0], [10, 0], [10, 10]])

    @pytest.mark.unit
    def test_single_point(self):
        """Test polyline with single point."""
        points = [(5, 5)]
        result = polyline(points)

        assert result.shape == (1, 1, 2)
        np.testing.assert_array_equal(result[0, 0, :], [5, 5])

    @pytest.mark.unit
    def test_float_conversion(self):
        """Test that float coordinates are converted to int."""
        points = [(5.7, 8.3), (12.9, 15.1)]
        result = polyline(points)

        assert result.dtype == np.int32
        np.testing.assert_array_equal(result[:, 0, :], [[5, 8], [12, 15]])

    @pytest.mark.unit
    def test_large_polyline(self):
        """Test with many points."""
        points = [(i, i*2) for i in range(100)]
        result = polyline(points)

        assert result.shape == (100, 1, 2)
        assert result[0, 0, 0] == 0
        assert result[99, 0, 0] == 99
