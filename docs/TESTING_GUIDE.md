# Testing Guide

## Overview

This project uses pytest for comprehensive testing with a goal of 80%+ code coverage.

## Quick Start

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/unit/test_geo.py
```

### Run tests with coverage
```bash
pytest --cov=app --cov=tools --cov-report=html
```

### Run only fast tests
```bash
pytest -m "not slow"
```

### Run specific test markers
```bash
pytest -m unit           # Only unit tests
pytest -m integration    # Only integration tests
pytest -m regression     # Only regression tests
```

## Test Structure

### Directory Layout
```
tests/
├── conftest.py              # Shared fixtures
├── fixtures/                # Test data
│   ├── sample_data/
│   └── regression_baseline/
├── unit/                    # Unit tests (isolated)
│   ├── test_geo.py
│   ├── test_iou_tracker.py
│   └── test_event_engine.py
├── integration/             # Integration tests (multiple components)
└── regression/              # Regression tests (baseline comparison)
```

## Writing Tests

### Test Class Structure
```python
class TestFeatureName:
    """Tests for specific feature."""

    @pytest.mark.unit
    def test_normal_case(self):
        """Test description."""
        # Arrange
        input_data = create_test_data()

        # Act
        result = function_under_test(input_data)

        # Assert
        assert result == expected_output
```

### Best Practices

1. **Use descriptive names**
   ```python
   # Good
   def test_iou_returns_zero_for_non_overlapping_boxes(self):

   # Bad
   def test_iou(self):
   ```

2. **Test edge cases**
   - Empty input
   - None values
   - Negative numbers
   - Boundary conditions
   - Large inputs

3. **Use fixtures for reusable data**
   ```python
   @pytest.fixture
   def sample_bbox():
       return (100, 200, 150, 300)

   def test_with_fixture(sample_bbox):
       assert sample_bbox[0] == 100
   ```

4. **Use parametrize for multiple test cases**
   ```python
   @pytest.mark.parametrize("input,expected", [
       ((0, 0, 10, 10), 100),
       ((5, 5, 15, 15), 100),
   ])
   def test_area(input, expected):
       assert calculate_area(input) == expected
   ```

5. **Mark tests appropriately**
   ```python
   @pytest.mark.unit         # Fast, isolated test
   @pytest.mark.slow         # Takes > 5 seconds
   @pytest.mark.gpu          # Requires GPU
   @pytest.mark.integration  # Tests multiple components
   ```

## Using Fixtures

### Available Fixtures (from conftest.py)

- `tmp_test_dir`: Temporary test directory
- `sample_config`: Configuration dictionary
- `sample_frame_1080p`: Synthetic video frame
- `sample_tracks_dataframe`: Sample tracking data
- `sample_detections`: Sample detection results
- `mock_yolo_model`: Mock YOLO for GPU-free testing

### Example Usage
```python
def test_with_fixtures(tmp_test_dir, sample_config):
    config_path = tmp_test_dir / "config.yaml"
    # Use fixtures in test
```

## Coverage Reports

### Generate HTML Report
```bash
pytest --cov=app --cov=tools --cov-report=html
```

Open `htmlcov/index.html` in browser to view detailed coverage.

### Terminal Report
```bash
pytest --cov=app --cov=tools --cov-report=term-missing
```

Shows coverage with line numbers of missing coverage.

### Coverage Goals
- Week 2: 30%
- Week 4: 50%
- Week 6: 65%
- Week 8: 80%+

## Testing Patterns

### Testing Exceptions
```python
def test_raises_value_error():
    with pytest.raises(ValueError, match="Missing columns"):
        load_invalid_csv()
```

### Testing Approximate Values
```python
def test_iou_approximate():
    assert iou_xyxy(box_a, box_b) == pytest.approx(0.333, abs=0.01)
```

### Testing DataFrame Equality
```python
import pandas as pd
from pandas.testing import assert_frame_equal

def test_dataframe_processing():
    result_df = process_tracks(input_df)
    assert_frame_equal(result_df, expected_df)
```

### Mocking External Dependencies
```python
from unittest.mock import Mock, patch

def test_with_mock():
    with patch('app.module.yolo_model') as mock_yolo:
        mock_yolo.predict.return_value = mock_results
        result = detect_objects(frame)
        assert result is not None
```

## Integration Tests

Integration tests verify multiple components working together.

```python
@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline(tmp_test_dir, sample_tracks_dataframe):
    # Save tracks
    tracks_path = tmp_test_dir / "tracks.csv"
    sample_tracks_dataframe.to_csv(tracks_path, index=False)

    # Process through pipeline
    engine = EventEngine()
    tracks = engine.load_tracks(tracks_path)
    possessions, _, _ = engine.compute_possessions(tracks)

    # Verify output
    assert len(possessions) > 0
```

## Regression Tests

Regression tests compare current output against known baseline.

```python
@pytest.mark.regression
def test_tracks_count_unchanged(mac1_reference_tracks):
    current_tracks = run_tracking_pipeline("mac1.mp4")
    reference_tracks = pd.read_csv(mac1_reference_tracks)

    # Track count should be stable
    assert len(current_tracks) == pytest.approx(len(reference_tracks), rel=0.05)
```

## Continuous Integration

### Pre-commit Hooks

Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

Hooks run automatically before each commit:
- Code formatting (black)
- Linting (flake8)
- Fast unit tests

### GitHub Actions

Tests run automatically on:
- Every push to main/develop
- Every pull request

View results at: https://github.com/your-repo/actions

## Troubleshooting

### Tests fail with import errors
```bash
# Ensure you're in project root
cd C:\Users\user\Desktop\basket

# Install dependencies
pip install -r requirements.txt
```

### Coverage not working
```bash
# Install pytest-cov
pip install pytest-cov

# Run with coverage
pytest --cov=app
```

### Slow test execution
```bash
# Run tests in parallel
pip install pytest-xdist
pytest -n auto
```

### Skip slow tests during development
```bash
pytest -m "not slow"
```

## Common Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test
pytest tests/unit/test_geo.py::TestPointInPoly::test_point_inside_square

# Run tests matching pattern
pytest -k "test_iou"

# Show test durations
pytest --durations=10

# Stop on first failure
pytest -x

# Drop into debugger on failure
pytest --pdb

# Show print statements
pytest -s

# Coverage report
pytest --cov=app --cov-report=term-missing

# HTML coverage report
pytest --cov=app --cov-report=html
open htmlcov/index.html
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Coverage.py](https://coverage.readthedocs.io/)
- Project-specific: `AJAN_4_TESTING_SPEC.md`
