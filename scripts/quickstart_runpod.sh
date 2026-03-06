#!/bin/bash
# Quick Start - Deploy microCoder 3B to RunPod using CLI

set -e

echo "🚀 microCoder RunPod Quick Start"
echo "================================="
echo ""

# Step 1: Configure RunPod CLI
echo "📋 Step 1: Configure RunPod CLI"
echo ""
echo "Get your API key from: https://www.runpod.io/console/user/settings"
echo ""
read -p "Enter your RunPod API key: " API_KEY

if [ -z "$API_KEY" ]; then
    echo "❌ API key required!"
    exit 1
fi

echo ""
echo "Setting up RunPod CLI..."
runpodctl config --apiKey "$API_KEY"

echo "✅ Configuration saved!"
echo ""

# Step 2: Verify dataset
echo "📋 Step 2: Verifying dataset..."
if [ ! -f "data/codesearchnet_python.pt" ]; then
    echo "❌ Dataset not found!"
    echo ""
    echo "Please prepare the dataset first:"
    echo "  python scripts/prepare_data.py --source hf --dataset code_search_net --language python --output data/codesearchnet_python.pt"
    exit 1
fi

DATASET_SIZE=$(du -h data/codesearchnet_python.pt | cut -f1)
echo "✅ Dataset ready: $DATASET_SIZE"
echo ""

# Step 3: Instructions for web UI
echo "📋 Step 3: Create RunPod Instance"
echo ""
echo "⚠️  Pod creation requires using the web UI:"
echo ""
echo "1. Go to: https://www.runpod.io/console/gpu-cloud"
echo "2. Filter for: A100 80GB"
echo "3. Select: Spot instance (~\$0.60/hr)"
echo "4. Template: PyTorch 2.1.0"
echo "5. Volume: 50GB"
echo "6. Click 'Deploy'"
echo ""
read -p "Press Enter once you've created the pod..."
echo ""
read -p "Enter your Pod ID: " POD_ID

if [ -z "$POD_ID" ]; then
    echo "❌ Pod ID required!"
    exit 1
fi

echo ""
echo "✅ Pod ID: $POD_ID"
echo ""

# Step 4: Check pod status
echo "📋 Step 4: Checking pod status..."
if ! runpodctl get pod "$POD_ID" > /dev/null 2>&1; then
    echo "❌ Pod not found or not ready!"
    echo "Please wait a moment and try again."
    exit 1
fi
echo "✅ Pod is ready!"
echo ""

# Step 5: Package and upload
echo "📋 Step 5: Packaging project files..."
echo ""

# Create temporary directory
TEMP_DIR=$(mktemp -d)
echo "Creating upload package..."

# Copy necessary files
mkdir -p "$TEMP_DIR"/{src,scripts,configs,data,models/checkpoints,logs}
cp -r src/* "$TEMP_DIR/src/"
cp -r scripts/* "$TEMP_DIR/scripts/"
cp -r configs/* "$TEMP_DIR/configs/"
cp requirements.txt "$TEMP_DIR/"
cp data/codesearchnet_python.pt "$TEMP_DIR/data/"
cp data/codesearchnet_python_metadata.json "$TEMP_DIR/data/" 2>/dev/null || true
touch "$TEMP_DIR/models/checkpoints/.gitkeep"
touch "$TEMP_DIR/logs/.gitkeep"

# Create archive
cd "$TEMP_DIR"
tar -czf /tmp/microcoder_upload.tar.gz .
cd - > /dev/null

ARCHIVE_SIZE=$(du -h /tmp/microcoder_upload.tar.gz | cut -f1)
echo "✅ Package created: $ARCHIVE_SIZE"
echo ""

# Upload
echo "📤 Uploading to pod (this may take a few minutes)..."
echo ""
runpodctl send "$POD_ID" /tmp/microcoder_upload.tar.gz

echo ""
echo "✅ Upload complete!"
echo ""

# Step 6: Setup environment
echo "📋 Step 6: Setting up environment on pod..."
echo ""

# Extract and install
runpodctl exec "$POD_ID" bash << 'EOF'
cd /workspace
echo "Extracting files..."
tar -xzf microcoder_upload.tar.gz
rm microcoder_upload.tar.gz

echo "Installing dependencies..."
pip install -q torch tiktoken datasets pyyaml tqdm tensorboard wandb

echo "Making scripts executable..."
chmod +x scripts/*.sh scripts/*.py 2>/dev/null || true

echo ""
echo "✅ Setup complete!"
echo ""
echo "Project structure:"
ls -lh
echo ""
echo "Dataset:"
ls -lh data/
EOF

echo ""
echo "✅ Environment ready!"
echo ""

# Step 7: Start training
echo "📋 Step 7: Starting training..."
echo ""
read -p "Start training now? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 Launching training in tmux session..."
    
    runpodctl exec "$POD_ID" bash -c "cd /workspace && tmux new-session -d -s training './scripts/train_3b_runpod.sh'"
    
    sleep 2
    
    echo ""
    echo "✅ Training started!"
fi

# Cleanup
rm -rf "$TEMP_DIR"
rm /tmp/microcoder_upload.tar.gz

# Final instructions
echo ""
echo "=========================================="
echo "🎉 Deployment Complete!"
echo "=========================================="
echo ""
echo "📊 Your Training Session:"
echo "  Pod ID: $POD_ID"
echo "  Cost: ~\$0.60/hour"
echo "  Duration: ~15-18 hours"
echo "  Total: ~\$9-11"
echo ""
echo "📊 Monitor Training:"
echo "  runpodctl exec $POD_ID tail -f /workspace/logs/training_3b.log"
echo ""
echo "  # Or attach to tmux session:"
echo "  runpodctl exec $POD_ID bash"
echo "  tmux attach -t training"
echo ""
echo "  # Or use live monitor:"
echo "  runpodctl exec $POD_ID bash"
echo "  python scripts/monitor_training.py logs/training_3b.log"
echo ""
echo "📥 Download Results (when done):"
echo "  runpodctl receive $POD_ID /workspace/models/checkpoints/codesearchnet_3b.pt ./models/checkpoints/"
echo ""
echo "🛑 Stop Pod (when done):"
echo "  runpodctl stop pod $POD_ID"
echo ""
echo "📚 Full documentation:"
echo "  docs/RUNPOD_CLI.md"
echo ""
echo "Happy training! 🚀"
