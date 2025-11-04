# Quick Reference: Pod Deployment Commands

## Local Machine (Before Pod)

### Setup
```bash
# Test locally with Mac9
python -m app.run

# Test batch script
python process_batch.py --video mac9.mp4

# Validate results
python qc_checker.py runs/run_*/
```

### Push Code to GitHub
```bash
cd /path/to/basket/github_bundle
git add -A
git commit -m "Add batch processing and pod setup scripts"
git push
```

---

## Pod Machine (After SSH)

### Initial Setup (One-time)
```bash
# SSH into pod
ssh -i ~/.ssh/pod_key.pem ubuntu@POD_IP

# Clone and setup
cd /root
git clone https://github.com/YOUR_USERNAME/basket.git
cd basket/github_bundle

# Run automated setup
bash pod_setup.sh

# Wait 5-10 minutes for PyTorch installation...
```

### Process Videos

**Single video:**
```bash
python process_batch.py --video /data/mac9.mp4
```

**All videos in directory:**
```bash
python process_batch.py --dir /data/videos/
```

**With custom output directory:**
```bash
python process_batch.py --dir /data/videos/ --output /data/runs/
```

### Monitor Processing
```bash
# Watch logs in real-time
tail -f runs/run_*/log.txt

# Or check GPU usage
nvidia-smi -l 1  # Refresh every 1 second
```

---

## File Transfer (From Local Machine)

### Upload Videos
```bash
# Single video
scp -i ~/.ssh/pod_key.pem mac9.mp4 ubuntu@POD_IP:/data/videos/

# Multiple videos
scp -i ~/.ssh/pod_key.pem mac*.mp4 ubuntu@POD_IP:/data/videos/
```

### Download Results
```bash
# Entire run directory
scp -r -i ~/.ssh/pod_key.pem ubuntu@POD_IP:/data/runs/run_20240115_120000 .

# Just the metadata
scp -i ~/.ssh/pod_key.pem ubuntu@POD_IP:/data/runs/run_20240115_120000/metadata.json .

# Just the log
scp -i ~/.ssh/pod_key.pem ubuntu@POD_IP:/data/runs/run_20240115_120000/log.txt .
```

---

## Typical Workflow

```bash
# ============ LOCAL MACHINE ============

# 1. Prepare code locally
python -m app.run                      # Test with Mac9

# 2. Push to GitHub
git push

# 3. Start pod and wait for IP
# (provider dashboard or CLI)

# 4. SSH setup
ssh -i ~/.ssh/pod_key.pem ubuntu@POD_IP
# Then follow pod setup below...

# ============ POD MACHINE (in SSH session) ============

# 5. Clone and setup
cd /root
git clone https://github.com/YOUR_USERNAME/basket.git
cd basket/github_bundle
bash pod_setup.sh

# Wait for setup to complete...

# ============ LOCAL MACHINE (new terminal) ============

# 6. Upload videos while pod sets up
scp -i ~/.ssh/pod_key.pem mac*.mp4 ubuntu@POD_IP:/data/videos/

# ============ POD MACHINE (continue SSH) ============

# 7. Process videos
python process_batch.py --dir /data/videos/ --output /data/runs/

# Watch progress
tail -f runs/run_*/log.txt

# ============ LOCAL MACHINE ============

# 8. Download results (after processing complete)
scp -r -i ~/.ssh/pod_key.pem ubuntu@POD_IP:/data/runs/run_* .

# 9. Validate results
python qc_checker.py runs/run_20240115_120000/

# ============ Done! ============
```

---

## Files Created for Pod Deployment

### Scripts
- **process_batch.py** - Batch processing with metadata/logging
- **qc_checker.py** - Validation and quality control
- **pod_setup.sh** - Automated environment setup

### Documentation
- **POD_DEPLOYMENT.md** - Full deployment guide (30+ pages)
- **QUICK_REFERENCE.md** - This file

### Configuration
- **configs/salon.yaml** - Already tuned with optimal settings

---

## Common Issues & Fixes

### CUDA Not Available
```bash
# On pod, check GPU
nvidia-smi

# Reinstall PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

### Slow Processing (CPU fallback)
```bash
# Check logs
grep "CUDA" runs/run_*/log.txt

# Should say "CUDA: True" - if not, CUDA setup failed
```

### Track ID Explosion (>30 IDs)
Edit `configs/salon.yaml`:
```yaml
tracking:
  iou_threshold: 0.25    # Decrease to match more aggressively
  max_age: 50            # Increase to keep tracks longer
```

Then re-run processing.

---

## Time Estimates

| Task | Time |
|------|------|
| Pod setup (first time) | 25-30 min |
| Upload single video | 1-2 min |
| Process single video (5 min duration) | 5-10 min |
| Download single result | 1-2 min |
| QC validation | 1-2 min |
| **Total first video** | **~40 min** |
| **Subsequent videos** | **~10-15 min each** |

---

## Cost Estimate

**Tesla V100 (RunPod): ~$0.50/hour**

| Scenario | Time | Cost |
|----------|------|------|
| 10 videos (50 min) | 1 hour | ~$0.50 |
| 20 videos (100 min) | 2 hours | ~$1.00 |
| 100 videos (500 min) | 8-9 hours | ~$4-5 |

**Pro tip:** Process all videos in one continuous session to avoid setup costs

---

## Useful Pod Commands

```bash
# Check GPU memory and usage
nvidia-smi

# Check Python packages installed
pip list | grep -E "torch|ultralytics|pandas"

# Check disk usage
df -h

# Check if Python virtual environment is active
echo $VIRTUAL_ENV

# Get pod IP from cloud provider
hostname -I
```

---

## Accessing Pod Again

If you disconnect and need to reconnect:

```bash
# SSH back in
ssh -i ~/.ssh/pod_key.pem ubuntu@POD_IP

# Activate environment
cd /root/basket/github_bundle
source venv/bin/activate

# Check status
ls -la runs/run_*/
tail -f runs/run_*/log.txt
```

---

## Before Closing Pod

```bash
# Download all results
scp -r -i ~/.ssh/pod_key.pem ubuntu@POD_IP:/data/runs/* .

# Verify all files downloaded
ls runs/run_*/*.csv

# Stop pod (through provider dashboard or CLI)
# This stops billing
```

