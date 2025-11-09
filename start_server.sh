#!/bin/bash
# Simulation Control Panel Server Startup Script
# For use on cluster environments

# Set environment variables for GPU and headless rendering
export HABITAT_GPU_ID=${HABITAT_GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export EGL_DEVICE_ID=${EGL_DEVICE_ID:-0}
export DISPLAY=${DISPLAY:-:0}

echo "=================================="
echo "Starting Simulation Control Server"
echo "=================================="
echo "HABITAT_GPU_ID: $HABITAT_GPU_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "EGL_DEVICE_ID: $EGL_DEVICE_ID"
echo "DISPLAY: $DISPLAY"
echo "=================================="
echo ""
echo "Server will be available at: http://localhost:8000"
echo "Set up SSH port forwarding on your local machine:"
echo "  ssh -L 8000:localhost:8000 $(whoami)@$(hostname)"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================="
echo ""

# Start the FastAPI server with uvicorn
uvicorn server:app --host 0.0.0.0 --port 8000

