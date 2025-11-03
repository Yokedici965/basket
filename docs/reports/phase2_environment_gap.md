## Missing Runtime Dependencies
- [ ] opencv-python → Required for: frame extraction, hoop overlay rendering
- [ ] pytest + pytest-cov → Required for: test execution, coverage reports
- [ ] streamlit OR gradio → Required for: QC dashboard UI

## Missing Code Modules
- [ ] app/events/shot_classifier.py → Status: missing
- [ ] tests/test_hoop_locator.py → Status: missing
- [ ] tests/test_calibration_integration.py → Status: missing

## Workaround Strategy
- **For calibration tests:** Write mock tests using built-in unittest (no opencv/pytest needed)
- **For QC UI:** Build CLI-based review tool first (tools/calibration_review.py already exists, enhance it)
- **For benchmarks:** Use existing JSON outputs, calculate metrics from logs (no video processing needed)
