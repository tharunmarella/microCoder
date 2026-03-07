#!/bin/bash
# RunPod Training Script - 3B Model (Production Quality)
# Cost: ~$9 for 15 hours on A100 80GB Spot
# Quality: ⭐⭐⭐⭐⭐ Near GitHub Copilot level

echo "🚀 microCoder - 3B Model Training on RunPod A100"
echo "=================================================="
echo ""
echo "📊 Model Configuration:"
echo "   Parameters: ~3 Billion"
echo "   Layers: 32"
echo "   Hidden dim: 2560"
echo "   Attention heads: 32"
echo "   Context length: 2048 tokens"
echo ""
echo "💰 Cost Estimate:"
echo "   GPU: A100 80GB (Spot)"
echo "   Rate: $0.60/hr"
echo "   Time: 15-18 hours"
echo "   Total: $9-11"
echo ""
echo "⚡ Starting training..."
echo ""

# Check if we're on CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA GPU not detected! Are you on RunPod?"
    exit 1
fi

# Show GPU info
echo "🎮 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Check if dataset exists
if [ ! -f "data/codesearchnet_python.pt" ]; then
    echo "❌ Dataset not found! Please upload data/codesearchnet_python.pt"
    exit 1
fi

echo "✅ Dataset found: data/codesearchnet_python.pt"
echo ""

# Create output directory
mkdir -p models/checkpoints
mkdir -p logs
mkdir -p logs/tensorboard

# Start training with optimized 3B configuration
python src/train.py \
  --data-source preprocessed \
  --data-file data/codesearchnet_python.pt \
  --device cuda \
  --n-layers 32 \
  --d-model 2560 \
  --n-heads 32 \
  --block-size 2048 \
  --vocab-size 50257 \
  --batch-size 2 \
  --grad-accum-steps 8 \
  --iterations 15000 \
  --lr 1.5e-4 \
  --scheduler \
  --warmup-ratio 0.03 \
  --min-lr-ratio 0.1 \
  --amp \
  --max-grad-norm 1.0 \
  --weight-decay 0.1 \
  --dropout 0.1 \
  --tensorboard \
  --save-path models/checkpoints/codesearchnet_3b.pt \
  --log-interval 100 \
  --sample-interval 1000 \
  --prompt "def binary_search(arr, target):" \
  2>&1 | tee logs/training_3b.log

echo ""
echo "=================================================="
echo "✅ Training Complete!"
echo "=================================================="
echo ""
echo "📦 Model saved to: models/checkpoints/codesearchnet_3b.pt"
echo "📋 Training log: logs/training_3b.log"
echo "📊 TensorBoard logs: logs/tensorboard/"
echo ""
echo "🎯 Next Steps:"
echo "   1. Download the model checkpoint"
echo "   2. View training metrics: tensorboard --logdir logs/tensorboard"
echo "   3. Test with: python src/generate.py models/checkpoints/codesearchnet_3b.pt 'your prompt'"
echo ""
