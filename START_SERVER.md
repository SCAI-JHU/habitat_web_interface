# Running Simulation Control Panel on Cluster

## Prerequisites

Make sure you have activated your conda/virtual environment with all requirements installed.

## Step 1: Set Environment Variables for Headless Rendering

```bash
# Set these in your terminal before starting the server
export DISPLAY=:0  # Or leave unset for true headless
export HABITAT_GPU_ID=0  # Your GPU ID
export CUDA_VISIBLE_DEVICES=0  # Should match your GPU
export EGL_DEVICE_ID=0  # Should match your GPU
```

## Step 2: Start the FastAPI Server

From the project root directory:

```bash
# Option A: Using uvicorn directly
uvicorn server:app --host 0.0.0.0 --port 8000

# Option B: With auto-reload for development
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

The server will start and listen on port 8000.

## Step 3: Set Up SSH Port Forwarding (On Your Local Machine)

Open a new terminal on your **local machine** and run:

```bash
ssh -L 8000:localhost:8000 username@cluster-address
```

Replace:
- `username` with your cluster username
- `cluster-address` with your cluster hostname

This will forward port 8000 from the cluster to your local machine.

## Step 4: Access the Web Interface

Open your web browser on your local machine and go to:

```
http://localhost:8000
```

You should see the simulation control panel!

## Troubleshooting

### Issue: "Address already in use"

If port 8000 is already in use, change the port:

```bash
uvicorn server:app --host 0.0.0.0 --port 8888
```

And update your SSH port forward accordingly:
```bash
ssh -L 8888:localhost:8888 username@cluster-address
```

### Issue: Simulation fails to start or controls don't work

1. **Check logs**: Look at the terminal where uvicorn is running for error messages
2. **Check GPU**: Make sure `HABITAT_GPU_ID` matches an available GPU:
   ```bash
   nvidia-smi  # Check available GPUs
   ```
3. **Check DISPLAY**: For headless rendering, you might need:
   ```bash
   export DISPLAY=""  # Empty for EGL rendering
   ```

### Issue: WebSocket connection fails

Make sure you're accessing via `http://localhost:8000` (not the cluster hostname directly).
The WebSocket connection needs to go through your SSH tunnel.

## Quick Start Script

You can create a script `start_server.sh`:

```bash
#!/bin/bash
export HABITAT_GPU_ID=0
export CUDA_VISIBLE_DEVICES=0
export EGL_DEVICE_ID=0
export DISPLAY=:0

uvicorn server:app --host 0.0.0.0 --port 8000
```

Make it executable and run:
```bash
chmod +x start_server.sh
./start_server.sh
```

## Stopping the Server

Press `Ctrl+C` in the terminal where uvicorn is running.

