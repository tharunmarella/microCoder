#!/bin/bash
# RunPod CLI Setup and Deployment Script for microCoder 3B Training
# This script automates the entire RunPod workflow using runpodctl

set -e  # Exit on any error

echo "🚀 microCoder RunPod Deployment Script"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR=$(pwd)
DATASET_FILE="data/codesearchnet_python.pt"
DATASET_METADATA="data/codesearchnet_python_metadata.json"

# Step 1: Check if runpodctl is configured
echo -e "${BLUE}📋 Step 1: Checking RunPod CLI configuration...${NC}"
if ! runpodctl config 2>/dev/null | grep -q "API Key"; then
    echo -e "${RED}❌ RunPod CLI not configured!${NC}"
    echo ""
    echo "Please run: runpodctl config"
    echo "You'll need your API key from: https://www.runpod.io/console/user/settings"
    exit 1
fi
echo -e "${GREEN}✅ RunPod CLI configured${NC}"
echo ""

# Step 2: Check dataset exists
echo -e "${BLUE}📋 Step 2: Checking dataset...${NC}"
if [ ! -f "$DATASET_FILE" ]; then
    echo -e "${RED}❌ Dataset not found: $DATASET_FILE${NC}"
    echo ""
    echo "Please prepare the dataset first:"
    echo "  python scripts/prepare_data.py --source hf --dataset code_search_net --language python --output $DATASET_FILE"
    exit 1
fi

DATASET_SIZE=$(du -h "$DATASET_FILE" | cut -f1)
echo -e "${GREEN}✅ Dataset found: $DATASET_SIZE${NC}"
echo ""

# Step 3: Find available A100 80GB pods
echo -e "${BLUE}📋 Step 3: Finding available A100 80GB GPUs...${NC}"
echo ""
echo "Searching for A100 80GB Spot instances..."
echo ""

# Note: runpodctl doesn't have a built-in search, so we'll guide the user
echo -e "${YELLOW}To find available pods, visit:${NC}"
echo "  https://www.runpod.io/console/gpu-cloud"
echo ""
echo "Look for:"
echo "  • GPU: A100 80GB"
echo "  • Type: Spot (for cost savings)"
echo "  • Storage: 50GB+"
echo "  • Cost: ~$0.60/hr"
echo ""
read -p "Press Enter when you've found an available pod and are ready to continue..."
echo ""

# Step 4: Create pod configuration
echo -e "${BLUE}📋 Step 4: Creating pod configuration...${NC}"

cat > /tmp/runpod_config.json <<EOF
{
  "name": "microCoder-3B-Training",
  "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
  "gpuTypeId": "NVIDIA A100 80GB PCIe",
  "cloudType": "SPOT",
  "supportPublicIp": true,
  "volumeInGb": 50,
  "containerDiskInGb": 50,
  "env": [
    {
      "key": "JUPYTER_PASSWORD",
      "value": "microcoder123"
    }
  ]
}
EOF

echo -e "${GREEN}✅ Configuration created${NC}"
echo ""

# Step 5: Deploy pod
echo -e "${BLUE}📋 Step 5: Deploying pod...${NC}"
echo ""
echo -e "${YELLOW}Note: This will start charging approximately $0.60/hour${NC}"
read -p "Continue with deployment? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

echo ""
echo "Creating pod..."
echo ""

# Create the pod using runpodctl
# Note: Actual deployment requires going through the web UI or API
# runpodctl is mainly for managing existing pods
echo -e "${YELLOW}⚠️  Pod creation via CLI requires additional setup.${NC}"
echo ""
echo "Please create the pod via web UI with these settings:"
echo "  • Name: microCoder-3B-Training"
echo "  • GPU: A100 80GB"
echo "  • Template: PyTorch 2.1.0"
echo "  • Storage: 50GB"
echo "  • Type: Spot"
echo ""
echo "Once created, you'll get a Pod ID."
read -p "Enter your Pod ID (e.g., abc123xyz): " POD_ID
echo ""

# Step 6: Wait for pod to be ready
echo -e "${BLUE}📋 Step 6: Waiting for pod to be ready...${NC}"
echo ""
echo "Checking pod status..."
# runpodctl get pod $POD_ID

# Step 7: Upload project files
echo -e "${BLUE}📋 Step 7: Uploading project files...${NC}"
echo ""

# Create a temporary directory with only necessary files
echo "Preparing files for upload..."
UPLOAD_DIR="/tmp/microcoder_upload"
rm -rf "$UPLOAD_DIR"
mkdir -p "$UPLOAD_DIR"

# Copy necessary files
echo "  • Copying source code..."
cp -r src "$UPLOAD_DIR/"
cp -r scripts "$UPLOAD_DIR/"
cp -r configs "$UPLOAD_DIR/"
mkdir -p "$UPLOAD_DIR/models/checkpoints"
mkdir -p "$UPLOAD_DIR/logs"
touch "$UPLOAD_DIR/models/checkpoints/.gitkeep"
touch "$UPLOAD_DIR/logs/.gitkeep"

echo "  • Copying requirements.txt..."
cp requirements.txt "$UPLOAD_DIR/"

echo "  • Copying dataset (this may take a while)..."
mkdir -p "$UPLOAD_DIR/data"
cp "$DATASET_FILE" "$UPLOAD_DIR/data/"
cp "$DATASET_METADATA" "$UPLOAD_DIR/data/"

echo ""
echo -e "${GREEN}✅ Files prepared for upload${NC}"
echo ""

# Create upload archive
echo "Creating upload archive..."
cd "$UPLOAD_DIR"
tar -czf /tmp/microcoder.tar.gz .
cd "$PROJECT_DIR"

ARCHIVE_SIZE=$(du -h /tmp/microcoder.tar.gz | cut -f1)
echo -e "${GREEN}✅ Archive created: $ARCHIVE_SIZE${NC}"
echo ""

# Upload to pod
echo "Uploading to pod $POD_ID..."
echo -e "${YELLOW}Note: This will take several minutes depending on your connection.${NC}"
echo ""

# Using runpodctl send command
runpodctl send /tmp/microcoder.tar.gz "$POD_ID":/workspace/

echo ""
echo -e "${GREEN}✅ Upload complete!${NC}"
echo ""

# Step 8: Connect and setup
echo -e "${BLUE}📋 Step 8: Setting up environment on pod...${NC}"
echo ""

# Get SSH connection command
SSH_COMMAND=$(runpodctl get pod "$POD_ID" 2>/dev/null | grep -i "ssh" || echo "")

echo "Connecting to pod to set up environment..."
echo ""
echo "You can connect with:"
echo "  runpodctl exec $POD_ID bash"
echo ""

# Create setup script for the pod
cat > /tmp/setup_pod.sh <<'PODSCRIPT'
#!/bin/bash
# This script runs on the RunPod instance

echo "🔧 Setting up microCoder environment..."

# Extract uploaded files
cd /workspace
tar -xzf microcoder.tar.gz
rm microcoder.tar.gz

# Install dependencies
pip install -q torch tiktoken datasets pyyaml tqdm

# Make scripts executable
chmod +x scripts/*.sh scripts/*.py

echo "✅ Setup complete!"
echo ""
echo "📊 Project structure:"
ls -lh

echo ""
echo "📦 Dataset:"
ls -lh data/*.pt

echo ""
echo "🎯 Ready to train! Run:"
echo "  ./scripts/train_3b_runpod.sh"
PODSCRIPT

chmod +x /tmp/setup_pod.sh

# Copy setup script to pod
runpodctl send /tmp/setup_pod.sh "$POD_ID":/workspace/

# Execute setup script
echo "Running setup on pod..."
runpodctl exec "$POD_ID" bash /workspace/setup_pod.sh

echo ""
echo -e "${GREEN}✅ Environment setup complete!${NC}"
echo ""

# Step 9: Start training
echo -e "${BLUE}📋 Step 9: Starting training...${NC}"
echo ""
read -p "Start training now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 Launching training..."
    echo ""
    
    # Start training in background with tmux
    runpodctl exec "$POD_ID" bash -c "cd /workspace && tmux new-session -d -s training './scripts/train_3b_runpod.sh'"
    
    echo -e "${GREEN}✅ Training started in tmux session 'training'${NC}"
    echo ""
fi

# Final instructions
echo ""
echo "=========================================="
echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo "=========================================="
echo ""
echo "Pod ID: $POD_ID"
echo "Cost: ~$0.60/hour"
echo "Training time: ~15-18 hours"
echo "Total cost: ~$9-11"
echo ""
echo "📊 Monitor Training:"
echo "  # Connect to pod"
echo "  runpodctl exec $POD_ID bash"
echo ""
echo "  # Attach to training session"
echo "  tmux attach -t training"
echo ""
echo "  # View logs"
echo "  tail -f /workspace/logs/training_3b.log"
echo ""
echo "  # Monitor GPU"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "📥 Download Results (after training):"
echo "  runpodctl receive $POD_ID /workspace/models/checkpoints/codesearchnet_3b.pt ./models/checkpoints/"
echo ""
echo "🛑 Stop Pod (when done):"
echo "  runpodctl stop pod $POD_ID"
echo ""
echo "💾 Cleanup local files:"
echo "  rm /tmp/microcoder.tar.gz"
echo "  rm -rf $UPLOAD_DIR"
echo ""

# Cleanup
rm -rf "$UPLOAD_DIR"
rm /tmp/microcoder.tar.gz
rm /tmp/setup_pod.sh
rm /tmp/runpod_config.json

echo -e "${GREEN}Happy training! 🚀${NC}"
