#!/bin/bash
# Open TensorBoard visualization

echo "🎨 Opening TensorBoard..."
echo ""
echo "Starting SSH port forward..."
echo "TensorBoard will open at: http://localhost:6006"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Port forward and open browser
ssh -L 6006:localhost:6006 -p 15717 root@64.247.201.48 -N &
SSH_PID=$!

sleep 2
open http://localhost:6006 2>/dev/null || echo "Open your browser to: http://localhost:6006"

# Wait for user to press Ctrl+C
trap "kill $SSH_PID; echo ''; echo '👋 TensorBoard closed'; exit" INT
wait
