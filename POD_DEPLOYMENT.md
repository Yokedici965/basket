# Pod Deployment Guide

## Overview

This guide explains how to deploy the basketball video analysis system to a remote GPU pod (Tesla V100) for high-speed batch processing.

**Hardware:** Tesla V100 32GB, Ubuntu 22.04 LTS, NVIDIA Driver 550
**Processing Time:** ~2-5 min per video (vs ~12 min on CPU)
**Cost:** Pod rental fees based on usage

---

## Pre-Deployment Checklist

Before opening the pod, verify:

- [ ] Code pushed to GitHub: `git push`
- [ ] Local testing with Mac9 completed: `python -m app.run`
- [ ] Config optimized in `configs/salon.yaml`
- [ ] `process_batch.py` created locally and tested
- [ ] `qc_checker.py` created locally and tested
- [ ] Pod credentials obtained (IP, username, SSH key)

---

## Phase 1: Pod Setup (30 minutes)

### Step 1: Open Pod

1. Log into your pod provider (RunPod, Lambda Labs, etc.)
2. Select Tesla V100 32GB instance
3. Choose Ubuntu 22.04 LTS base image
4. Note the public IP address

### Step 2: SSH Connection

```bash
# From your local machine
# Adjust IP and key path as needed
ssh -i /path/to/key.pem ubuntu@POD_IP

# If using password auth
ssh ubuntu@POD_IP
```

### Step 3: Clone Code and Setup

Once SSH'd into pod:

```bash
# Clone repository
cd /root
git clone https://github.com/YOUR_USERNAME/basket.git
cd basket/github_bundle

# Run setup script (automated environment)
bash pod_setup.sh

# This will:
# - Create Python virtual environment
# - Install PyTorch with CUDA 12.1
# - Install YOLOv8 and dependencies
# - Verify GPU access
```

### Step 4: Verify Setup

```bash
# Activate environment
source venv/bin/activate

# Test CUDA
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Quick test on small video (if available)
python -m app.run
```

---

## Phase 2: Data Transfer & Processing (variable)

### Option A: Single Video Processing

```bash
# From pod
python process_batch.py --video /path/to/video.mp4

# Output goes to runs/run_TIMESTAMP/
# Check results
ls runs/run_*/
cat runs/run_*/log.txt
```

### Option B: Batch Processing

```bash
# Create video directory on pod
mkdir -p /data/videos
mkdir -p /data/runs

# From your local machine, upload videos
scp -i /path/to/key.pem mac9.mp4 ubuntu@POD_IP:/data/videos/
scp -i /path/to/key.pem mac8.mp4 ubuntu@POD_IP:/data/videos/
scp -i /path/to/key.pem mac7.mp4 ubuntu@POD_IP:/data/videos/

# On pod, process all videos
cd /root/basket/github_bundle
python process_batch.py --dir /data/videos/ --output /data/runs/
```

### File Transfer Commands

**Upload video to pod:**
```bash
scp -i ~/.ssh/pod_key.pem local_video.mp4 ubuntu@POD_IP:/data/videos/
```

**Download results from pod:**
```bash
# Download entire run
scp -r -i ~/.ssh/pod_key.pem ubuntu@POD_IP:/data/runs/run_20240115_120000 .

# Download specific CSV
scp -i ~/.ssh/pod_key.pem ubuntu@POD_IP:/data/runs/run_20240115_120000/mac9_tracks.csv .
```

**Monitor processing on pod:**
```bash
# SSH into pod
ssh -i ~/.ssh/pod_key.pem ubuntu@POD_IP

# Watch progress (from pod)
tail -f basket/github_bundle/runs/run_*/log.txt
```

---

## Phase 3: Local QC Validation (5 minutes)

After downloading results to your local machine:

```bash
# Validate entire run
python qc_checker.py runs/run_20240115_120000/

# Output shows:
# - Track ID counts (should be 8-15 for basketball)
# - Confidence statistics
# - Data quality issues
# - Summary report
```

**Expected Output for Mac9:**
```
✅ mac9_MP4.mp4
   Tracks CSV: mac9_MP4_tracks.csv
     • Total rows: 30,060
     • Unique track IDs: 8-15 (GOOD - was 366 with old config!)
     • Person tracks: 5-6
     • Ball tracks: 1-2
     • Conf: min=0.250, max=0.999, mean=0.850
     • Static ball: 45 flagged
```

---

## Workflow Timeline

### First Run (Total: ~1 hour)

```
[Local] Code ready                    5 min
[Pod]   Setup environment             25 min
[Pod]   Process Mac9 (1:16 video)    5-10 min
[Local] Download results              5 min
[Local] QC validation                 5 min
[Local] Analyze & iterate             10 min
```

### Subsequent Runs (Much faster)

```
[Local] Upload new video(s)          5 min
[Pod]   Process videos               5-10 min per video
[Local] Download & validate          5 min
Total: ~15-20 min per video
```

---

## Configuration Tuning

If QC shows issues, adjust `configs/salon.yaml` on pod:

```yaml
detection:
  base_conf: 0.15        # YOLO confidence threshold
  nms_iou: 0.45          # Non-max suppression
  thresholds:
    ball: 0.04           # Lower = more detections (may be noisier)
    person: 0.25

tracking:
  iou_threshold: 0.3     # Lower = stricter matching (fewer ID switches)
  max_age: 40            # Frames to keep lost tracks (higher = better with occlusion)
  center_distance_threshold: 60.0  # Max center distance for matching
```

**For Track ID Explosion:**
- Decrease `iou_threshold` (0.3 → 0.25) to match more aggressively
- Increase `max_age` (40 → 50) to keep tracks through occlusion
- Decrease `center_distance_threshold` to match only nearby objects

**For Ball Detection Issues:**
- Decrease `ball` threshold (0.04 → 0.03) to catch more detections
- Try smaller model: `yolov8s.pt` instead of `yolov8l.pt`

---

## Troubleshooting

### GPU Not Available on Pod

```bash
# Check CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

### Slow Processing

1. Check if GPU is being used:
   ```bash
   watch -n 1 'nvidia-smi'
   ```
   Should show GPU memory usage during processing

2. If CPU-bound, CUDA failed silently - check drivers:
   ```bash
   nvidia-smi
   ```

### Out of Memory

For 32GB V100, you should be able to process 4K video with `img_size: 1024`. If OOM:
- Reduce `img_size` to 832 or 640
- Process with `frame_stride: 3` or higher (skip more frames)
- Process one video at a time (not parallel)

### Pod Disconnection

If SSH drops during processing:
```bash
# Processing continues on pod, reconnect and check
ssh ubuntu@POD_IP
tail -f runs/run_*/log.txt
```

---

## Cost Estimation

### Tesla V100 32GB

**RunPod Pricing (~$0.50/hour):**
- 10 videos × 5 min = 50 min = ~$0.42
- 100 videos × 5 min = 500 min = $4.17

**Lambda Labs (~$0.60/hour):**
- 10 videos = ~$0.50
- 100 videos = $5.00

**Tips to Minimize Cost:**
1. Process multiple videos in one session
2. Don't keep pod idle between sessions
3. Use smaller model if accuracy permits (`yolov8s.pt`)
4. Test locally first before pod deployment

---

## Output Files Reference

Each run produces:

```
runs/run_20240115_120000/
├── metadata.json           # Configuration used, timestamps, results summary
├── log.txt                 # Detailed processing log
├── mac9_MP4_tracks.csv     # Tracked objects (person/ball with track_id)
├── mac9_MP4_detections.csv # Raw detections (before tracking)
├── mac8_MP4_tracks.csv
└── mac8_MP4_detections.csv
```

**tracks.csv columns:**
```
frame, ts, cls, track_id, conf, x1, y1, x2, y2, cx, cy, w, h, age, hits, is_static, is_predicted, is_ref
```

**detections.csv columns:**
```
frame, ts, cls, conf, x1, y1, x2, y2, cx, cy, is_static
```

---

## Next Steps After Validation

1. **If Track IDs Good (8-15):** Proceed to event motor implementation
2. **If Track IDs Bad (>30):** Adjust config and re-run
3. **If Ball Unstable:** Try lower ball threshold or yolov8s model
4. **If Everything Good:** Process full video dataset

---

## FAQ

**Q: Can I process multiple videos in parallel?**
A: Not currently. `process_batch.py` processes sequentially but quickly due to GPU speed.

**Q: How long until results?**
A: ~5-10 min per video on V100, vs ~12 min on CPU

**Q: Can I keep the pod running between sessions?**
A: Yes, but you're charged for idle time. Better to upload/process/download then stop pod.

**Q: What if I need to adjust config?**
A: Edit `configs/salon.yaml` locally, push to GitHub, pull on pod, re-run

**Q: Can I use a cheaper GPU?**
A: V100 is good balance. A100 costs more, RTX 4090 is overkill for this task. T4/L4 would be slower.

