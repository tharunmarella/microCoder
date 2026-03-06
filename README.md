# microCoder 🚀

A minimal, high-performance GPT-like LLM for **code generation**, built with modern best practices.

<div align="center">

**100M parameters • PyTorch • RoPE • Flash Attention • RMSNorm**

</div>

---

## ✨ Features

### Modern Architecture
- **RMSNorm** (Llama-style) - More stable than LayerNorm
- **Rotary Positional Embeddings (RoPE)** - Better position encoding than learned embeddings
- **Flash Attention** - 2-4x faster attention computation
- **Weight Tying** - Reduces parameters by ~26%
- **GELU Activation** - Standard for transformers

### Training Optimizations
- **GPT-style Initialization** - Proper weight scaling for deep networks
- **Warmup + Cosine LR Schedule** - Smooth learning rate decay
- **AdamW Optimizer** - Weight decay for better generalization
- **Gradient Clipping** - Prevents exploding gradients
- **Mixed Precision (AMP)** - 2x speedup on modern GPUs
- **Gradient Accumulation** - Simulate larger batch sizes

### Production Features
- **Modular Architecture** - Clean separation of concerns
- **Preprocessed Datasets** - Load data once, train many times
- **Multiple Model Sizes** - From 10M to 1.3B parameters
- **MPS Support** - Optimized for Apple Silicon (M1/M2/M3/M4)
- **Comprehensive Logging** - Track training progress

---

## 📁 Project Structure

```
microCoder/
├── src/                     # Source code
│   ├── __init__.py
│   ├── model.py            # Model architecture
│   ├── data.py             # Data loading utilities
│   ├── utils.py            # Helper functions
│   ├── train.py            # Training script
│   └── generate.py         # Text generation
├── scripts/                 # Utility scripts
│   └── prepare_data.py     # Dataset preprocessing
├── configs/                 # Model configurations
│   └── model_configs.yaml  # Model size presets
├── data/                    # Preprocessed datasets
├── models/                  # Trained models
│   ├── checkpoints/        # Model checkpoints
│   └── exported_weights/   # Exported weights
├── logs/                    # Training logs
└── README.md
```

---

## 🛠️ Installation

```bash
# Clone the repository
git clone <repository-url>
cd microCoder

# Install dependencies
pip install torch tiktoken datasets

# Optional: For YAML config loading
pip install pyyaml
```

---

## 🚀 Quick Start

### 1. Prepare Dataset (One-time)

```bash
# Prepare Python code dataset
python scripts/prepare_data.py \
    --source hf \
    --dataset flytech/python-codes-25k \
    --output data/python_codes_25k.pt
```

**Other datasets:**
```bash
# The Stack (Python subset)
python scripts/prepare_data.py \
    --dataset bigcode/the-stack \
    --config data/python \
    --output data/the_stack_python.pt

# Your own code
python scripts/prepare_data.py \
    --source file \
    --file your_code.txt \
    --output data/custom.pt
```

### 2. Train a Model

**100M parameter model** (recommended for M1/M2/M3/M4 Macs):
```bash
python src/train.py \
    --data-source preprocessed \
    --data-file data/python_codes_25k.pt \
    --device mps \
    --n-layers 10 \
    --d-model 768 \
    --n-heads 12 \
    --block-size 512 \
    --batch-size 8 \
    --iterations 2000 \
    --scheduler \
    --save-path models/checkpoints/model_100m.pt
```

**For NVIDIA GPUs:**
```bash
python src/train.py \
    --data-source preprocessed \
    --data-file data/python_codes_25k.pt \
    --device cuda \
    --n-layers 10 \
    --d-model 768 \
    --n-heads 12 \
    --block-size 512 \
    --batch-size 8 \
    --amp \
    --iterations 5000 \
    --scheduler \
    --save-path models/checkpoints/model_100m.pt
```

### 3. Generate Code

```bash
python src/generate.py \
    models/checkpoints/model_100m.pt \
    "def fibonacci(n):" \
    --max-new 100 \
    --temperature 0.8
```

---

## 📊 Model Sizes

| Size | Parameters | Layers | Hidden Dim | VRAM | Best For |
|------|------------|--------|------------|------|----------|
| **Tiny** | 10M | 6 | 384 | ~2 GB | Testing |
| **Small** | 100M | 10 | 768 | ~6 GB | M1/M2/M3/M4 Macs |
| **Medium** | 350M | 12 | 1024 | ~16 GB | Gaming GPUs |
| **Large** | 800M | 16 | 1536 | ~32 GB | Professional GPUs |
| **XL** | 1.3B | 24 | 2048 | ~48 GB | A100/H100 |

---

## 🎯 Training Arguments

### Model Architecture
- `--vocab-size` - Vocabulary size (default: 50257 for GPT-2)
- `--block-size` - Context length (default: 512)
- `--n-layers` - Number of transformer layers (default: 10)
- `--d-model` - Hidden dimension (default: 768)
- `--n-heads` - Number of attention heads (default: 12)
- `--dropout` - Dropout rate (default: 0.1)

### Training
- `--iterations` - Number of training steps (default: 2000)
- `--batch-size` - Batch size (default: 8)
- `--lr` - Learning rate (default: 3e-4)
- `--scheduler` - Use warmup + cosine LR schedule
- `--warmup-ratio` - Warmup ratio (default: 0.02 = 2%)
- `--amp` - Use mixed precision training
- `--grad-accum-steps` - Gradient accumulation steps (default: 2)

### Data
- `--data-source` - Data source: `preprocessed`, `hf`, `file`, or `synthetic`
- `--data-file` - Path to preprocessed data or text file

### System
- `--device` - Device: `cuda`, `mps`, or `cpu`
- `--save-path` - Where to save the trained model

---

## 🧠 Architecture Details

### Transformer Block
```
Input → RMSNorm → Multi-Head Attention (with RoPE) → Add & Norm
                                                          ↓
      ← Feed-Forward Network (MLP) ← RMSNorm ← ───────────
                   ↓
              Output
```

### Key Components
- **RoPE (Rotary Positional Embeddings)**: Encodes position information by rotating query/key vectors
- **Flash Attention**: PyTorch's optimized attention kernel (up to 4x faster)
- **RMSNorm**: Normalizes by root mean square (more stable than LayerNorm)
- **Weight Tying**: Shares weights between embedding and output layers

---

## 📈 Performance Tips

### For Mac M1/M2/M3/M4:
- Use `--device mps` for Apple Silicon acceleration
- Start with the "small" config (100M params)
- Use `--batch-size 6-8` for optimal memory usage
- Don't use `--amp` (MPS handles precision automatically)

### For NVIDIA GPUs:
- Use `--device cuda` and `--amp` for 2x speedup
- Increase `--batch-size` if you have more VRAM
- Use `--grad-accum-steps` to simulate larger batches

### For Training Speed:
- Preprocess datasets once with `prepare_data.py`
- Use `--scheduler` for better convergence
- Enable `--amp` on CUDA for faster training

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📝 License

MIT License - feel free to use for your own projects!

---

## 🙏 Acknowledgments

Architecture inspired by:
- **GPT-2/GPT-3** (OpenAI) - Overall architecture
- **Llama** (Meta) - RMSNorm, RoPE
- **Flash Attention** (Tri Dao) - Efficient attention
- **nanoGPT** (Andrej Karpathy) - Clean implementation style

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{microcoder2024,
  title = {microCoder: A Minimal GPT-like LLM for Code Generation},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/microCoder}
}
```
