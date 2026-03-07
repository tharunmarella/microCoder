# microCoder Project Structure

Clean, organized LLM training codebase for Python code generation models.

## 📁 Directory Structure

```
microCoder/
├── src/                    # Core library code
│   ├── train.py           # Main training script with GPT architecture
│   ├── model.py           # Model architecture (Transformer, Attention, etc.)
│   ├── data.py            # Data loading and tokenization
│   ├── generate.py        # Text generation/inference
│   └── utils.py           # Helper functions
│
├── scripts/               # Training and deployment scripts
│   ├── prepare_data.py    # Dataset preprocessing
│   ├── download_dataset.sh # Download datasets from HuggingFace
│   ├── train_3b_runpod.sh # 3B model training script for RunPod
│   ├── test_model_runpod.py # Model testing on RunPod
│   ├── quickstart_runpod.sh # Automated RunPod deployment
│   ├── deploy_runpod.sh   # Detailed RunPod deployment script
│   ├── setup_pod.sh       # SSH-based pod setup
│   ├── runpod_commands.sh # RunPod CLI reference
│   ├── monitor_training.py # Training monitoring tools
│   ├── visualize_training.py # Live GPU/training visualization
│   ├── tensorboard_logger.py # TensorBoard integration
│   ├── wandb_setup.py     # Weights & Biases setup
│   └── inspect_dataset.py # Dataset inspection utilities
│
├── docs/                  # Documentation
│   ├── RUNPOD_SETUP.md   # RunPod deployment guide
│   ├── RUNPOD_CLI.md     # RunPod CLI usage and automation
│   ├── MONITORING.md     # Training monitoring guide
│   ├── MONITORING_QUICKSTART.md # Quick monitoring reference
│   └── INSTRUCTION_FINETUNING.md # Instruction fine-tuning guide
│
├── configs/              # Training configurations
│   └── train_config.yaml # Training hyperparameters
│
├── data/                 # Training data (gitignored)
│   ├── .gitkeep
│   └── codesearchnet_python.pt (353MB, preprocessed)
│
├── models/               # Model checkpoints (gitignored)
│   ├── checkpoints/
│   │   └── .gitkeep
│   └── exported_weights/
│       └── .gitkeep
│
├── logs/                 # Training logs (gitignored)
│   ├── .gitkeep
│   └── tensorboard/      # TensorBoard logs
│
├── README.md            # Project overview and quick start
├── requirements.txt     # Python dependencies
└── .gitignore          # Git ignore rules
```

## 🎯 Key Files

### Core Training
- **`src/train.py`** - Main training script with:
  - GPT-like architecture (Flash Attention, RoPE, RMSNorm)
  - AdamW optimizer with gradient clipping
  - Mixed precision training (AMP)
  - Learning rate warmup + cosine decay
  - TensorBoard integration
  - Weight initialization (GPT-style)

### Model Architecture
- **`src/model.py`** - Transformer implementation:
  - Multi-head self-attention
  - Rotary positional embeddings (RoPE)
  - RMSNorm (instead of LayerNorm)
  - GELU activation
  - Weight tying (input/output embeddings)

### Data Processing
- **`src/data.py`** - Data handling:
  - HuggingFace dataset loading
  - tiktoken (GPT-2 BPE) tokenization
  - Byte-level fallback
  - Preprocessed data loading
  - Custom DataLoader

### Generation
- **`src/generate.py`** - Inference:
  - Top-k sampling
  - Temperature control
  - Checkpoint loading
  - Multiple prompts support

## 🚀 Quick Start

### 1. Local Setup
```bash
pip install -r requirements.txt
python scripts/prepare_data.py
```

### 2. RunPod Training (Recommended)
```bash
# Automated deployment
./scripts/quickstart_runpod.sh

# Or manual
runpodctl start gpu --name microcoder-training \
  --gpuType "NVIDIA H100 SXM" --volumeSize 50
```

### 3. Test Model
```bash
# On RunPod
ssh -p <PORT> root@<IP> 'cd /workspace && python scripts/test_model_runpod.py'

# Or locally
python src/generate.py models/checkpoints/model.pt "def binary_search(arr, target):"
```

## 📊 Training Configurations

### 100M Model (Testing)
- Layers: 12, Hidden: 768, Heads: 12
- Context: 512 tokens
- Cost: ~$0.50, Time: ~30 min on A40
- Quality: Basic code completion

### 3B Model (Production)
- Layers: 32, Hidden: 2560, Heads: 32
- Context: 2048 tokens
- Cost: ~$2-3, Time: ~1 hour on H100
- Quality: Comparable to CodeGen-3B/StarCoder-3B

### 7B Model (High Quality)
- Layers: 32, Hidden: 4096, Heads: 32
- Context: 4096 tokens
- Cost: ~$18-24, Time: ~6-8 hours on H100
- Quality: Comparable to CodeLlama-7B

## 📈 Monitoring

### TensorBoard (Recommended)
```bash
# Start TensorBoard
tensorboard --logdir logs/tensorboard --host 0.0.0.0 --port 6006

# With SSH port forward
ssh -L 6006:localhost:6006 -p <PORT> root@<IP>
# Open: http://localhost:6006
```

### Live Monitor
```bash
# Real-time GPU/training stats
python scripts/visualize_training.py
```

### Weights & Biases
```bash
# Cloud-based tracking
python scripts/wandb_setup.py
python src/train.py --wandb
```

## 🛠️ Development

### Adding New Features
1. Core logic → `src/`
2. Scripts → `scripts/`
3. Documentation → `docs/`
4. Tests → `scripts/test_*.py`

### Training New Model
1. Edit `configs/train_config.yaml`
2. Or pass args: `python src/train.py --n-layers 24 --d-model 2048`
3. Monitor with TensorBoard
4. Test with `src/generate.py`

### Fine-tuning for Instructions
See `docs/INSTRUCTION_FINETUNING.md` for:
- Supervised Fine-Tuning (SFT)
- Dataset options (Code Alpaca, WizardCoder)
- Configuration changes
- Expected outcomes

## 📦 Dependencies

Core:
- `torch>=2.0.0` - PyTorch
- `tiktoken` - OpenAI's BPE tokenizer
- `datasets` - HuggingFace datasets
- `pyyaml` - Config files
- `tqdm` - Progress bars

Optional:
- `tensorboard` - Training visualization
- `wandb` - Cloud experiment tracking
- `flash-attn` - Faster attention (CUDA only)

## 🧹 Cleanup Commands

```bash
# Remove cache files
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name ".DS_Store" -delete

# Clear logs
rm -rf logs/*.log logs/*.txt

# Clear checkpoints
rm -rf models/checkpoints/*.pt
```

## 🔗 External Resources

- RunPod: https://runpod.io
- HuggingFace: https://huggingface.co/datasets/code_search_net
- TensorBoard: https://www.tensorflow.org/tensorboard
- W&B: https://wandb.ai

---

**Last Updated:** March 2026
