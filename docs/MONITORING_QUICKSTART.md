# Training Data Visualization Tools - Quick Reference

## 🎯 Best Options (Ranked)

### 1. **Live Monitor** 📊 (Start here!)
```bash
python scripts/monitor_training.py logs/training_3b.log
```
- ✅ **Zero setup** - works immediately
- ✅ Real-time dashboard with progress bars
- ✅ Loss trends with sparklines
- ✅ ETA and speed metrics
- ✅ Perfect for quick checks

### 2. **TensorBoard** 📈 (For analysis)
```bash
pip install tensorboard
tensorboard --logdir logs/tensorboard
# Open: http://localhost:6006
```
- ✅ Beautiful graphs and charts
- ✅ Weight/gradient histograms
- ✅ Compare multiple runs
- ✅ Works offline

### 3. **Weights & Biases** ☁️ (For production)
```bash
pip install wandb
wandb login
# Monitor from anywhere!
```
- ✅ Cloud-based (access from phone!)
- ✅ Team collaboration
- ✅ Best UI/UX
- ✅ Free for personal use

---

## Quick Commands

```bash
# Basic log viewing
tail -f logs/training_3b.log

# Live dashboard (recommended)
python scripts/monitor_training.py logs/training_3b.log

# TensorBoard
tensorboard --logdir logs/tensorboard

# Check GPU usage (RunPod)
watch -n 1 nvidia-smi
```

---

## Full Documentation
See `docs/MONITORING.md` for:
- Detailed setup guides
- Integration examples
- Use case recommendations
- Tips and tricks
