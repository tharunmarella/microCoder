#!/bin/bash
# Manual setup script for RunPod - Use this after uploading files

set -e

POD_ID="w7q36vtw8s573q"

echo "🔧 Setting up microCoder on RunPod"
echo "==================================="
echo ""
echo "Pod ID: $POD_ID"
echo ""

# Get SSH connection
echo "📡 Getting SSH connection info..."
SSH_CMD=$(runpodctl ssh connect "$POD_ID")
echo "SSH: $SSH_CMD"
echo ""

# Extract host and port
SSH_HOST=$(echo $SSH_CMD | sed 's/ssh root@//' | cut -d' ' -f1)
SSH_PORT=$(echo $SSH_CMD | sed 's/.*-p //')

echo "Host: $SSH_HOST"
echo "Port: $SSH_PORT"
echo ""

# Run setup commands via SSH
echo "📋 Setting up environment..."
echo ""

ssh -p $SSH_PORT root@$SSH_HOST << 'EOF'
set -e

cd /workspace

echo "📦 Extracting files..."
if [ -f microcoder_upload.tar.gz ]; then
    tar -xzf microcoder_upload.tar.gz
    rm microcoder_upload.tar.gz
    echo "✅ Files extracted"
else
    echo "❌ Upload file not found!"
    exit 1
fi

echo ""
echo "📚 Installing dependencies..."
pip install -q torch tiktoken datasets pyyaml tqdm tensorboard wandb

echo ""
echo "🔨 Making scripts executable..."
chmod +x scripts/*.sh scripts/*.py 2>/dev/null || true

echo ""
echo "✅ Setup complete!"
echo ""
echo "📊 Project structure:"
ls -lh

echo ""
echo "📦 Dataset:"
ls -lh data/

echo ""
echo "🎯 Ready to train!"
EOF

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "🚀 To start training, connect to the pod:"
echo "  $SSH_CMD"
echo ""
echo "Then run:"
echo "  cd /workspace"
echo "  tmux new -s training"
echo "  ./scripts/train_3b_runpod.sh"
echo ""
echo "Or start training directly:"
echo "  ssh -p $SSH_PORT root@$SSH_HOST 'cd /workspace && tmux new-session -d -s training \"./scripts/train_3b_runpod.sh\"'"
echo ""
