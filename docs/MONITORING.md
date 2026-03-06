# Training Monitoring Tools

microCoder includes multiple options for monitoring your training progress:

## 🎯 Quick Comparison

| Tool | Best For | Setup | Cost | Access |
|------|----------|-------|------|--------|
| **Basic Log** | Quick checks | None | Free | Local |
| **Live Monitor** | Real-time CLI | None | Free | Local |
| **TensorBoard** | Detailed analysis | `pip install tensorboard` | Free | Local |
| **W&B** | Cloud + Collaboration | `pip install wandb` | Free* | Anywhere |

*Free for personal projects

---

## 1. Basic Log Monitoring (Built-in)

**No setup required!** Training automatically logs to `logs/training_3b.log`

```bash
# Watch log in real-time
tail -f logs/training_3b.log

# Check last 50 lines
tail -n 50 logs/training_3b.log

# Search for specific iteration
grep "Iteration 1000" logs/training_3b.log
```

**Example output:**
```
Iteration 100/15000 | Loss: 3.456 | Time: 1.23s | LR: 0.00015
Iteration 200/15000 | Loss: 3.234 | Time: 1.21s | LR: 0.00015
```

---

## 2. Live Monitor Dashboard 📊 (Recommended)

**Interactive CLI dashboard** with progress bars, sparklines, and ETA!

```bash
# Start live monitor (auto-refreshes every 10s)
python scripts/monitor_training.py logs/training_3b.log

# Or just check current stats once
python scripts/monitor_training.py logs/training_3b.log --once
```

**Features:**
- ✅ Real-time progress bar
- ✅ Loss trend sparkline (📈📉)
- ✅ ETA estimation
- ✅ Speed metrics (iterations/sec)
- ✅ Recent history
- ✅ No installation needed!

**Screenshot:**
```
======================================================================
📊 microCoder Training Monitor - Live
======================================================================
Log: logs/training_3b.log
Updated: 18:30:45
======================================================================

🔄 Iteration: 1,234 / 15,000
   Progress: [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 8.2%

📉 Loss: 2.8453
   Perplexity: 17.21
   Trend: ▇▆▅▅▄▄▃▃▃▂▂▂▂▁▁▁▁▁▁▁ 📉 (-0.0234)

⚡ Speed: 1.23s per iteration
   ETA: 4.7h
   Est. completion: 2026-03-06 23:12

📚 Learning Rate: 1.48e-04

📊 Recent History:
   Iter  1,230: Loss 2.8687 | Time 1.21s
   Iter  1,231: Loss 2.8621 | Time 1.22s
   Iter  1,232: Loss 2.8554 | Time 1.24s
   Iter  1,233: Loss 2.8502 | Time 1.23s
   Iter  1,234: Loss 2.8453 | Time 1.23s

======================================================================

⏱️  Refreshing in 10s... (Ctrl+C to stop)
```

---

## 3. TensorBoard 📈

**Detailed visualizations** with graphs, histograms, and more!

### Setup:
```bash
pip install tensorboard
```

### Usage:
1. **During training**, logs are saved to `logs/tensorboard/`
2. **Start TensorBoard:**
   ```bash
   tensorboard --logdir logs/tensorboard --port 6006
   ```
3. **Open browser:** http://localhost:6006

### Features:
- ✅ Loss/perplexity graphs over time
- ✅ Learning rate schedule visualization
- ✅ Weight & gradient histograms
- ✅ Model architecture graph
- ✅ Generated text samples
- ✅ Compare multiple runs

### Integration:
```python
# Add to your training script:
from scripts.tensorboard_logger import TrainingMonitor

monitor = TrainingMonitor(experiment_name="3b_codesearchnet")

# In training loop:
monitor.log_metrics({
    "train/loss": loss.item(),
    "train/perplexity": math.exp(loss.item()),
    "train/lr": lr
}, step=iteration)
```

See `scripts/tensorboard_logger.py` for full integration guide.

---

## 4. Weights & Biases (W&B) ☁️ (Best for Production)

**Cloud-based tracking** - Monitor from anywhere, share with team!

### Setup:
```bash
pip install wandb
wandb login  # One-time setup
```

### Usage:
```python
import wandb

# Initialize at start of training
wandb.init(
    project="microCoder",
    name="3b-codesearchnet",
    config={"n_layers": 32, "d_model": 2560, ...}
)

# Log during training
wandb.log({
    "loss": loss.item(),
    "learning_rate": lr,
    "iteration": iteration
})

# Finish at end
wandb.finish()
```

### Features:
- ✅ Access from anywhere (web + mobile app!)
- ✅ Beautiful interactive dashboards
- ✅ Automatic hyperparameter tracking
- ✅ Easy sharing (public or private)
- ✅ Model versioning & artifacts
- ✅ Compare runs side-by-side
- ✅ Free for personal projects

### View your runs:
`https://wandb.ai/<your-username>/microCoder`

See `scripts/wandb_setup.py` for full integration guide.

---

## 5. RunPod Monitoring

When training on RunPod, you can also use:

### GPU Usage:
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or more detailed:
watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv'
```

### Process Monitoring:
```bash
# Check if training is running
ps aux | grep python

# Monitor CPU/memory
htop  # or 'top'
```

### Remote Access to Logs:
```bash
# From your local machine, sync logs:
rsync -avz runpod:/workspace/microCoder/logs/ ./logs/

# Then use local monitoring tools
python scripts/monitor_training.py logs/training_3b.log
```

---

## Recommendations by Use Case

### Local Development (Mac M4):
- **Best:** Live Monitor + Basic logs
- **Why:** Lightweight, no setup needed

### Local GPU Training:
- **Best:** TensorBoard + Live Monitor
- **Why:** Detailed analysis without cloud dependency

### Cloud Training (RunPod):
- **Best:** W&B + Live Monitor
- **Why:** Monitor from anywhere, great for long training runs

### Team/Collaboration:
- **Best:** W&B
- **Why:** Easy sharing, commenting, and comparison

### Research/Experimentation:
- **Best:** TensorBoard or W&B
- **Why:** Detailed metrics, hyperparameter tracking

---

## Quick Start Examples

### 1. Minimal (Just logs):
```bash
# Start training
./scripts/train_3b_runpod.sh | tee logs/training_3b.log

# Monitor in another terminal
tail -f logs/training_3b.log
```

### 2. Interactive (Live Monitor):
```bash
# Terminal 1: Start training
./scripts/train_3b_runpod.sh

# Terminal 2: Start monitor
python scripts/monitor_training.py logs/training_3b.log
```

### 3. Full Featured (TensorBoard):
```bash
# Terminal 1: Start training with TensorBoard integration
python src/train.py --tensorboard ...

# Terminal 2: Start TensorBoard
tensorboard --logdir logs/tensorboard

# Terminal 3: Live monitor
python scripts/monitor_training.py logs/training_3b.log
```

### 4. Cloud (W&B):
```bash
# One-time setup
pip install wandb
wandb login

# Start training with W&B
python src/train.py --wandb ...

# Monitor from browser or mobile app!
# https://wandb.ai/<username>/microCoder
```

---

## Tips

1. **Use multiple tools!** Live Monitor for quick checks + TensorBoard/W&B for analysis
2. **Save logs!** Always use `tee` or redirect output: `./train.sh 2>&1 | tee logs/training.log`
3. **Check early!** Monitor first few iterations to catch issues fast
4. **Plot learning curves** to detect overfitting or training instability
5. **Set alerts** (W&B supports Slack/email alerts for anomalies)

---

Need help? Check the integration guides:
- `scripts/tensorboard_logger.py` - TensorBoard integration
- `scripts/wandb_setup.py` - W&B integration
- `scripts/monitor_training.py` - Live monitor usage
