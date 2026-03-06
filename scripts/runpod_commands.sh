#!/bin/bash
# Quick RunPod CLI Commands Reference for microCoder

echo "🎯 RunPod CLI Quick Reference for microCoder"
echo "============================================="
echo ""

# Check configuration
echo "📋 CHECK CONFIGURATION:"
echo "  runpodctl config"
echo "  # Set API key from: https://www.runpod.io/console/user/settings"
echo ""

# List available GPU types and pricing
echo "💰 CHECK PRICING:"
echo "  # Visit RunPod console to see current pricing"
echo "  # https://www.runpod.io/console/gpu-cloud"
echo "  # A100 80GB Spot: ~\$0.60/hr"
echo ""

# List your pods
echo "📦 LIST YOUR PODS:"
echo "  runpodctl get pod"
echo ""

# Get specific pod info
echo "ℹ️  GET POD INFO:"
echo "  runpodctl get pod <POD_ID>"
echo ""

# Execute commands on pod
echo "🖥️  EXECUTE COMMANDS:"
echo "  runpodctl exec <POD_ID> <command>"
echo "  runpodctl exec <POD_ID> bash"
echo "  runpodctl exec <POD_ID> nvidia-smi"
echo ""

# Upload files
echo "⬆️  UPLOAD FILES:"
echo "  runpodctl send <local_file> <POD_ID>:<remote_path>"
echo "  runpodctl send data/codesearchnet_python.pt <POD_ID>:/workspace/data/"
echo ""

# Download files
echo "⬇️  DOWNLOAD FILES:"
echo "  runpodctl receive <POD_ID> <remote_file> <local_path>"
echo "  runpodctl receive <POD_ID> /workspace/models/checkpoints/codesearchnet_3b.pt ./models/checkpoints/"
echo ""

# Stop pod
echo "🛑 STOP POD:"
echo "  runpodctl stop pod <POD_ID>"
echo ""

# Remove pod
echo "🗑️  REMOVE POD:"
echo "  runpodctl remove pod <POD_ID>"
echo ""

# SSH into pod (if SSH enabled)
echo "🔐 SSH INTO POD:"
echo "  # Get SSH command"
echo "  runpodctl get pod <POD_ID> | grep ssh"
echo "  # Or use exec"
echo "  runpodctl exec <POD_ID> bash"
echo ""

echo "========================================="
echo "📚 Full documentation:"
echo "  https://docs.runpod.io/cli/install-runpodctl"
echo ""
