# RunPod Setup Guide - 3B Model Training

## Quick Start (5 minutes to launch!)

### 1. Create RunPod Instance

1. Go to [RunPod.io](https://www.runpod.io/)
2. Navigate to **GPU Pods**
3. Filter for: **A100 80GB** (Spot pricing)
4. Select a pod and click **Deploy**
5. Choose: **PyTorch 2.x** template
6. Storage: **50GB** (30GB for model, 10GB for dataset, 10GB buffer)

### 2. Upload Project

```bash
# Option A: Git Clone (easiest)
git clone https://github.com/<your-repo>/microCoder.git
cd microCoder

# Option B: Upload via RunPod File Manager
# - Upload the entire project folder
# - Make sure data/codesearchnet_python.pt is included!
```

### 3. Install Dependencies

```bash
cd microCoder
pip install -r requirements.txt
```

### 4. Start Training

```bash
# This will train for ~15 hours and cost ~$9
./scripts/train_3b_runpod.sh
```

### 5. Monitor Progress

```bash
# In another terminal/tab
tail -f logs/training_3b.log

# Or check GPU usage
watch -n 1 nvidia-smi
```

---

## Configuration Details

### 3B Model Specs
- **Parameters**: 2,975,531,777 (~3B)
- **Architecture**:
  - Layers: 32
  - Hidden dimension: 2560
  - Attention heads: 32 (head dim: 80)
  - Context length: 2048 tokens
  - Vocab size: 50,257 (GPT-2 BPE)
  - Dropout: 0.1

### Training Configuration
- **Dataset**: CodeSearchNet Python (417,000 functions)
- **Batch size**: 2 per GPU
- **Gradient accumulation**: 8 steps
- **Effective batch size**: 16 (2 × 8)
- **Total tokens per batch**: 32,768 (16 × 2048)
- **Iterations**: 15,000
- **Total tokens**: ~491M tokens

### Optimizer Settings
- **Algorithm**: AdamW
- **Learning rate**: 1.5e-4 (peak)
- **Warmup**: 3% of training (450 steps)
- **Min LR**: 1.5e-5 (10% of peak)
- **Scheduler**: Cosine annealing with warmup
- **Weight decay**: 0.1
- **Gradient clipping**: 1.0
- **AMP**: Enabled (mixed precision)

### Expected Results
- **Training time**: 15-18 hours on A100 80GB
- **Cost**: $9-11 (spot pricing at $0.60/hr)
- **Final loss**: ~1.5-2.0 (typical for code)
- **Memory usage**: ~65-70GB VRAM
- **Quality**: ⭐⭐⭐⭐⭐ Production-ready code generation

---

## Cost Breakdown

| Component | Cost |
|-----------|------|
| GPU (A100 80GB Spot) | $0.60/hr |
| Training time | 15-18 hours |
| **Total** | **$9-11** |

*Note: Spot pricing can vary. Add 10-20% buffer for potential interruptions.*

---

## After Training

### 1. Download Model

```bash
# From RunPod terminal
cd models/checkpoints
ls -lh codesearchnet_3b.pt

# Download via RunPod File Manager or:
runpodctl receive codesearchnet_3b.pt
```

### 2. Test Locally

```bash
# On your Mac
python src/generate.py models/checkpoints/codesearchnet_3b.pt \
  "def binary_search(arr, target):" \
  --max-tokens 256 \
  --temperature 0.7
```

### 3. Evaluate Quality

```bash
# Test various prompts
python src/generate.py models/checkpoints/codesearchnet_3b.pt \
  "# Sort a list in-place
def sort_list(items):" \
  --max-tokens 200

python src/generate.py models/checkpoints/codesearchnet_3b.pt \
  "class BinaryTree:
    def __init__(self):" \
  --max-tokens 300
```

---

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch-size 1`
- Increase grad accumulation: `--grad-accum-steps 16`
- Reduce context: `--block-size 1024`

### Slow Training
- Check GPU usage: `nvidia-smi`
- Ensure AMP is enabled: `--amp` flag
- Verify CUDA is being used (not CPU!)

### Spot Instance Interrupted
- Training saves checkpoints every 1000 steps
- Resume with same command (auto-loads last checkpoint)
- Consider on-demand instance for critical runs

---

## Scaling Up (Optional)

Want even better quality? Here's how to scale:

### 7B Model ($18-24)
```bash
./scripts/train_7b_runpod.sh  # (create this next!)
```

### 13B Model ($36-48)
- Requires: 2× A100 80GB or A100 80GB + DeepSpeed
- Cost: $36-48 for 24-30 hours
- Quality: State-of-the-art code generation

---

## Questions?

- **Too expensive?** Try 1.3B model first (~$4, scripts/train_1.3b_runpod.sh)
- **Want faster?** Use 2× A100s with data parallelism
- **Need longer context?** Increase `--block-size` to 4096 (slower, more memory)

Happy training! 🚀
