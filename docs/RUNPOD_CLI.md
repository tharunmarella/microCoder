# RunPod CLI Deployment Guide for microCoder

Complete guide for deploying and training the 3B model on RunPod using the CLI.

---

## 🚀 Quick Start (3 Steps)

### 1. Configure RunPod CLI
```bash
# Set your API key (one-time setup)
runpodctl config

# Get your API key from:
# https://www.runpod.io/console/user/settings
```

### 2. Deploy with Automated Script
```bash
# Run the automated deployment script
./scripts/deploy_runpod.sh
```

This script will:
- ✅ Check your dataset is ready
- ✅ Find available A100 80GB pods
- ✅ Create and upload project files
- ✅ Set up the environment
- ✅ Start training automatically

### 3. Monitor Training
```bash
# Get your pod ID from the deployment output
export POD_ID="your-pod-id"

# Connect to pod
runpodctl exec $POD_ID bash

# View training progress
tail -f /workspace/logs/training_3b.log
```

---

## 📋 Manual Step-by-Step (Alternative)

If you prefer manual control:

### Step 1: Check CLI Configuration
```bash
runpodctl config
# Should show your API key
```

### Step 2: Create Pod via Web UI
Since `runpodctl` doesn't support pod creation directly, create via web:

1. Go to: https://www.runpod.io/console/gpu-cloud
2. Select: **A100 80GB** (Spot)
3. Template: **PyTorch 2.1.0**
4. Storage: **50GB**
5. Click **Deploy**
6. Copy your **Pod ID**

### Step 3: Check Pod Status
```bash
# View your pods
runpodctl get pod

# Check specific pod
runpodctl get pod <POD_ID>
```

### Step 4: Upload Project Files
```bash
# Create upload package
tar -czf microcoder.tar.gz \
    src/ \
    scripts/ \
    configs/ \
    requirements.txt \
    data/codesearchnet_python.pt \
    data/codesearchnet_python_metadata.json

# Upload to pod
runpodctl send <POD_ID> microcoder.tar.gz /workspace/

# Verify upload
runpodctl exec <POD_ID> ls -lh /workspace/
```

### Step 5: Set Up Environment
```bash
# Connect to pod
runpodctl exec <POD_ID> bash

# Extract files
cd /workspace
tar -xzf microcoder.tar.gz

# Install dependencies
pip install torch tiktoken datasets pyyaml tqdm

# Make scripts executable
chmod +x scripts/*.sh scripts/*.py

# Verify setup
ls -lh data/*.pt
```

### Step 6: Start Training
```bash
# Start training in tmux (so it continues if disconnected)
runpodctl exec <POD_ID> bash -c "cd /workspace && tmux new-session -d -s training './scripts/train_3b_runpod.sh'"

# Or start directly (will stop if disconnected)
runpodctl exec <POD_ID> bash -c "cd /workspace && ./scripts/train_3b_runpod.sh"
```

### Step 7: Monitor Training

**Option A: Live Monitor Dashboard**
```bash
# Connect to pod
runpodctl exec <POD_ID> bash

# Run live monitor
cd /workspace
python scripts/monitor_training.py logs/training_3b.log
```

**Option B: View Logs**
```bash
# Real-time log viewing
runpodctl exec <POD_ID> tail -f /workspace/logs/training_3b.log

# Check last 50 lines
runpodctl exec <POD_ID> tail -n 50 /workspace/logs/training_3b.log
```

**Option C: GPU Monitoring**
```bash
# Watch GPU usage
runpodctl exec <POD_ID> watch -n 1 nvidia-smi

# One-time check
runpodctl exec <POD_ID> nvidia-smi
```

**Option D: Attach to tmux session**
```bash
# Connect to pod
runpodctl exec <POD_ID> bash

# Attach to training session
tmux attach -t training

# Detach: Press Ctrl+B, then D
```

### Step 8: Download Results
```bash
# After training completes (15-18 hours)

# Download model checkpoint
runpodctl receive <POD_ID> \
    /workspace/models/checkpoints/codesearchnet_3b.pt \
    ./models/checkpoints/

# Download training log
runpodctl receive <POD_ID> \
    /workspace/logs/training_3b.log \
    ./logs/

# Verify download
ls -lh models/checkpoints/codesearchnet_3b.pt
```

### Step 9: Stop Pod
```bash
# Stop pod to stop charges
runpodctl stop pod <POD_ID>

# Or remove completely
runpodctl remove pod <POD_ID>
```

---

## 🎯 Common Commands

### Pod Management
```bash
# List all your pods
runpodctl get pod

# Get pod details
runpodctl get pod <POD_ID>

# Stop pod
runpodctl stop pod <POD_ID>

# Start stopped pod
runpodctl start pod <POD_ID>

# Remove pod
runpodctl remove pod <POD_ID>
```

### File Operations
```bash
# Upload single file
runpodctl send <POD_ID> local_file.txt /workspace/

# Upload directory (must tar first)
tar -czf data.tar.gz data/
runpodctl send <POD_ID> data.tar.gz /workspace/

# Download file
runpodctl receive <POD_ID> /workspace/model.pt ./models/

# List files on pod
runpodctl exec <POD_ID> ls -lh /workspace/
```

### Execute Commands
```bash
# Run single command
runpodctl exec <POD_ID> nvidia-smi

# Interactive bash session
runpodctl exec <POD_ID> bash

# Run command in specific directory
runpodctl exec <POD_ID> bash -c "cd /workspace && python --version"

# Check training status
runpodctl exec <POD_ID> ps aux | grep python
```

---

## 💡 Pro Tips

### 1. Use tmux for Long-Running Processes
```bash
# Start training in tmux
runpodctl exec <POD_ID> bash -c "tmux new-session -d -s training 'cd /workspace && ./scripts/train_3b_runpod.sh'"

# Check tmux sessions
runpodctl exec <POD_ID> tmux ls

# Attach to session
runpodctl exec <POD_ID> bash
tmux attach -t training

# Detach: Ctrl+B, then D
```

### 2. Download Logs Periodically
```bash
# Download logs while training is running (every hour)
while true; do
    runpodctl receive <POD_ID> /workspace/logs/training_3b.log ./logs/
    sleep 3600
done
```

### 3. Monitor from Local Machine
```bash
# Download log and monitor locally
runpodctl receive <POD_ID> /workspace/logs/training_3b.log ./logs/
python scripts/monitor_training.py logs/training_3b.log --once

# Repeat every 5 minutes
watch -n 300 'runpodctl receive <POD_ID> /workspace/logs/training_3b.log ./logs/ && python scripts/monitor_training.py logs/training_3b.log --once'
```

### 4. Check Training Progress
```bash
# Quick status check
runpodctl exec <POD_ID> bash -c "cd /workspace && tail -n 5 logs/training_3b.log"

# See how long it's been running
runpodctl exec <POD_ID> bash -c "ps aux | grep 'train_3b_runpod.sh' | grep -v grep"
```

### 5. Save Costs with Spot Instances
- Always use **Spot** instances (60-80% cheaper)
- Spot instances can be interrupted (rare for A100s)
- Training saves checkpoints every 1000 steps
- Can resume from checkpoint if interrupted

---

## 🔧 Troubleshooting

### Pod Not Found
```bash
# List all pods
runpodctl get pod

# Check if pod is stopped
runpodctl start pod <POD_ID>
```

### Upload Failed
```bash
# Check pod storage space
runpodctl exec <POD_ID> df -h

# Check file size before upload
ls -lh data/codesearchnet_python.pt

# Upload in chunks if needed
split -b 1G data/codesearchnet_python.pt data_part_
for part in data_part_*; do
    runpodctl send <POD_ID> $part /workspace/
done
```

### Training Not Starting
```bash
# Check if dependencies installed
runpodctl exec <POD_ID> pip list | grep torch

# Check if scripts are executable
runpodctl exec <POD_ID> ls -lh /workspace/scripts/

# Check CUDA availability
runpodctl exec <POD_ID> python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory
```bash
# Check GPU memory
runpodctl exec <POD_ID> nvidia-smi

# Kill stuck processes
runpodctl exec <POD_ID> pkill -9 python

# Restart training with smaller batch size
# Edit train_3b_runpod.sh and change --batch-size 2 to --batch-size 1
```

---

## 📊 Cost Estimation

| Component | Cost |
|-----------|------|
| Pod creation | Free |
| A100 80GB Spot | $0.60/hour |
| Storage (50GB) | $0.10/month |
| Training time | 15-18 hours |
| **Total** | **$9-11** |

### Cost-Saving Tips:
1. ✅ Use Spot instances (vs On-Demand: $2.40/hr)
2. ✅ Stop pod immediately after training
3. ✅ Download results and delete pod
4. ✅ Monitor training to catch issues early

---

## 📚 Resources

- **RunPod CLI Docs**: https://docs.runpod.io/cli/install-runpodctl
- **RunPod Console**: https://www.runpod.io/console
- **API Keys**: https://www.runpod.io/console/user/settings
- **Community Discord**: https://discord.gg/runpod

---

## 🎯 Quick Commands Cheat Sheet

```bash
# Setup
runpodctl config

# Deploy (automated)
./scripts/deploy_runpod.sh

# Monitor
runpodctl exec <POD_ID> tail -f /workspace/logs/training_3b.log

# Download results
runpodctl receive <POD_ID> /workspace/models/checkpoints/codesearchnet_3b.pt ./models/checkpoints/

# Stop
runpodctl stop pod <POD_ID>
```

---

**Ready to train?** Run `./scripts/deploy_runpod.sh` to get started! 🚀
