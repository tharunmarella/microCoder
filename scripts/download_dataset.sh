#!/bin/bash
# Dataset Download & Preparation Script
# Choose your dataset based on time and quality needs

echo "🎯 microCoder - Dataset Preparation Helper"
echo "=========================================="
echo ""
echo "Select a dataset to download:"
echo ""
echo "1. Code Alpaca (FASTEST - 20 mins total)"
echo "   - Size: ~100 MB"
echo "   - Best for: Chat-style code assistant"
echo "   - Training time: 1-2 hours"
echo ""
echo "2. The Stack - Python Subset (BEST QUALITY - 2-3 hours prep)"
echo "   - Size: 20-30 GB (can limit with --max-samples)"
echo "   - Best for: Production-grade code completion"
echo "   - Training time: 6-8 hours"
echo ""
echo "3. CodeParrot - Python (BALANCED - 30 mins prep)"
echo "   - Size: 5-10 GB"
echo "   - Best for: Good quality with reasonable size"
echo "   - Training time: 2-3 hours"
echo ""
echo "4. Current Dataset (Already prepared)"
echo "   - Size: 62 MB (8.1M tokens)"
echo "   - Best for: Quick testing"
echo "   - Training time: 30-60 mins"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
  1)
    echo ""
    echo "📦 Downloading Code Alpaca..."
    python scripts/prepare_data.py \
      --source hf \
      --dataset sahil2801/CodeAlpaca-20k \
      --output data/code_alpaca.pt \
      --vocab-size 50257
    echo ""
    echo "✅ Ready to train!"
    echo "Run: python src/train.py --data-source preprocessed --data-file data/code_alpaca.pt"
    ;;
  2)
    echo ""
    echo "📦 Downloading The Stack (Python)..."
    echo "⚠️  This will take 2-3 hours and use 20-30 GB"
    read -p "Continue? (y/n): " confirm
    if [ "$confirm" = "y" ]; then
      python scripts/prepare_data.py \
        --source hf \
        --dataset bigcode/the-stack \
        --config data/python \
        --output data/the_stack_python.pt \
        --vocab-size 50257
      echo ""
      echo "✅ Ready to train!"
      echo "Run: python src/train.py --data-source preprocessed --data-file data/the_stack_python.pt"
    fi
    ;;
  3)
    echo ""
    echo "📦 Downloading CodeParrot (Python subset)..."
    echo "Limiting to 100K samples for reasonable size"
    python scripts/prepare_data.py \
      --source hf \
      --dataset codeparrot/github-code \
      --config python \
      --output data/codeparrot_python.pt \
      --max-samples 100000 \
      --vocab-size 50257
    echo ""
    echo "✅ Ready to train!"
    echo "Run: python src/train.py --data-source preprocessed --data-file data/codeparrot_python.pt"
    ;;
  4)
    echo ""
    echo "✅ Using current dataset: data/python_codes_25k.pt"
    echo "Run: python src/train.py --data-source preprocessed --data-file data/python_codes_25k.pt"
    ;;
  *)
    echo "Invalid choice"
    ;;
esac
