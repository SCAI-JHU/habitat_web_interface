import os
import subprocess
import uuid
import asyncio # For handling subprocess streams
import sys # To find the python executable
import glob
import base64
import binascii
import datetime
import shutil
from pathlib import Path
from typing import Union, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request # Import WebSocket components
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Initialize the FastAPI app
app = FastAPI()

# Pydantic model for robot commands
class RobotCommand(BaseModel):
    command: str

# --- Global State Management ---
simulation_status = {
    "status": "idle",
    "message": "Ready to start",
}
# Keep track of connected WebSocket clients
connected_clients: list[WebSocket] = []
# Command queue for robot control (will be initialized in startup)
robot_command_queue: Union[asyncio.Queue, None] = None
simulation_process: Union[asyncio.subprocess.Process, None] = None


# --- Directory and Script Paths ---
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Check if React build exists, otherwise use web directory
WEB_DIST_DIR = os.path.join(PROJECT_ROOT_DIR, "web", "dist")
WEB_DIR = WEB_DIST_DIR if os.path.exists(WEB_DIST_DIR) else os.path.join(PROJECT_ROOT_DIR, "web")
SIMULATION_SCRIPT = "habitat_llm/examples/controllable_simulation.py"

# --- Live frame storage configuration ---
LIVE_FRAMES_BASE_DIR = Path(PROJECT_ROOT_DIR) / "data" / "live_frames"
LIVE_FRAMES_BASE_DIR.mkdir(parents=True, exist_ok=True)
LIVE_FRAMES_CURRENT_SYMLINK = LIVE_FRAMES_BASE_DIR / "current"
live_frame_session_dir: Optional[Path] = None
live_frame_counter = 0
live_frame_lock = asyncio.Lock()


def _reset_live_frame_directory():
    """
    Prepare a fresh directory to persist the current simulation's live frames.
    Also updates the `current` symlink so tooling/frontends can find the latest run.
    """
    global live_frame_session_dir, live_frame_counter

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    session_dir = LIVE_FRAMES_BASE_DIR / timestamp

    if session_dir.exists():
        shutil.rmtree(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    # Update "current" symlink atomically
    if LIVE_FRAMES_CURRENT_SYMLINK.exists() or LIVE_FRAMES_CURRENT_SYMLINK.is_symlink():
        LIVE_FRAMES_CURRENT_SYMLINK.unlink()
    LIVE_FRAMES_CURRENT_SYMLINK.symlink_to(session_dir, target_is_directory=True)

    live_frame_session_dir = session_dir
    live_frame_counter = 0
    print(f"[LIVE_FRAMES] Writing frames to {session_dir}")


async def _persist_frame_from_data_url(data_url: str):
    """
    Decode a data URL emitted by the simulation and persist it to disk.
    """
    if not data_url.startswith("data:image"):
        return

    global live_frame_session_dir, live_frame_counter
    if live_frame_session_dir is None:
        _reset_live_frame_directory()

    try:
        header, encoded = data_url.split(",", 1)
    except ValueError:
        print("[LIVE_FRAMES] Malformed data URL; skipping frame")
        return

    extension = "png"
    if "image/jpeg" in header or "image/jpg" in header:
        extension = "jpg"
    elif "image/webp" in header:
        extension = "webp"

    try:
        raw_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        print(f"[LIVE_FRAMES] Failed to decode frame: {exc}")
        return

    async with live_frame_lock:
        frame_index = live_frame_counter
        live_frame_counter += 1
        destination = live_frame_session_dir / f"frame_{frame_index:06d}.{extension}"

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, destination.write_bytes, raw_bytes)

# --- WebSocket Endpoint ---
@app.websocket("/ws/live_feed")
async def websocket_endpoint(websocket: WebSocket):
    """Handles WebSocket connections for the live image feed."""
    await websocket.accept()
    connected_clients.append(websocket)
    print(f"Client connected: {websocket.client}")
    try:
        # Keep the connection open indefinitely.
        # In a real app, you might add heartbeat messages.
        while True:
            # We just need to keep the connection open to receive broadcasts.
            # Using receive_text() with a long timeout or just a simple sleep.
            await asyncio.sleep(60) # Keep alive, check connection state periodically
    except WebSocketDisconnect:
        # Client closed the connection
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        print(f"Client disconnected: {websocket.client}")
    except Exception as e:
        # Handle other potential errors during the connection
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        print(f"WebSocket Error for client {websocket.client}: {e}")


# --- Function to Broadcast Messages to Clients ---
async def broadcast_message(message: str):
    """Sends a message to all currently connected WebSocket clients."""
    # Create a list of tasks for sending messages concurrently
    tasks = [client.send_text(message) for client in connected_clients]
    # Use gather to run sends concurrently and capture exceptions
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle clients that failed to send (likely disconnected)
    disconnected_clients = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # If sending failed, mark the client for removal
            client = connected_clients[i]
            disconnected_clients.append(client)
            print(f"Failed to send to client {client.client}, marking for removal: {result}")

    # Remove disconnected clients from the active list
    for client in disconnected_clients:
        if client in connected_clients:
            connected_clients.remove(client)
            print(f"Removed disconnected client: {client.client}")


# --- Image Streaming from Trajectory Directory ---
async def stream_trajectory_images(trajectory_base_dir):
    """Watch the trajectory directory and stream new images as they appear."""
    last_image_index = -1
    
    while simulation_status["status"] == "running":
        try:
            # Find the most recent epidx directory
            epidx_dirs = glob.glob(os.path.join(trajectory_base_dir, "epidx_*/main_agent/rgb"))
            if not epidx_dirs:
                await asyncio.sleep(1)
                continue
            
            # Get the most recent one
            rgb_dir = max(epidx_dirs, key=os.path.getmtime)
            
            # Find all jpg files
            jpg_files = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
            
            # Stream any new images
            for jpg_file in jpg_files:
                # Extract index from filename (e.g., "123.jpg" -> 123)
                idx = int(os.path.splitext(os.path.basename(jpg_file))[0])
                
                if idx > last_image_index:
                    # Read and encode the image
                    with open(jpg_file, 'rb') as f:
                        img_data = f.read()
                    b64_string = base64.b64encode(img_data).decode('utf-8')
                    
                    # Send via WebSocket
                    await broadcast_message(f"data:image/jpeg;base64,{b64_string}")
                    last_image_index = idx
                    
                    if idx % 10 == 0:  # Log every 10th image
                        print(f"[IMAGE_STREAM] Sent image {idx}")
            
            await asyncio.sleep(4)  # Check for new images twice per second
            
        except Exception as e:
            print(f"[IMAGE_STREAM] Error: {e}")
            await asyncio.sleep(1)

# --- Modified Simulation Runner (Runs as an async task) ---
async def run_simulation_process():
    """Runs the simulation script as a subprocess and streams its output via WebSocket."""
    global simulation_status

    simulation_status = {"status": "running", "message": "Simulation starting..."}
    await broadcast_message("status:running:Simulation starting...") # Send status update via WebSocket
    _reset_live_frame_directory()

    # Configure GPU environment variables for the subprocess
    env = os.environ.copy()
    habitat_gpu_id = os.environ.get("HABITAT_GPU_ID", "0")
    env["HABITAT_GPU_ID"] = habitat_gpu_id
    # Ensure CUDA and EGL devices match if not set (important for clusters)
    if "CUDA_VISIBLE_DEVICES" not in env: env["CUDA_VISIBLE_DEVICES"] = habitat_gpu_id
    if "EGL_DEVICE_ID" not in env: env["EGL_DEVICE_ID"] = habitat_gpu_id

    script_path = os.path.join(PROJECT_ROOT_DIR, SIMULATION_SCRIPT)
    if not os.path.exists(script_path):
        error_msg = f"Simulation script not found: {script_path}"
        print(f"[ERROR] {error_msg}")
        simulation_status = {"status": "error", "message": error_msg}
        await broadcast_message(f"status:error:{error_msg}")
        return
    
    print(f"[SERVER] Running habitat simulation script: {SIMULATION_SCRIPT}")
    print(f"[SERVER] Full path: {script_path}")
    print(f"[SERVER] GPU Configuration: CUDA={env.get('CUDA_VISIBLE_DEVICES')}, EGL={env.get('EGL_DEVICE_ID')}")

    global simulation_process
    process = None # Define process variable outside try block
    try:
        # Use asyncio's subprocess handling for non-blocking I/O
        # Use absolute path to ensure we're running the right script
        script_abs_path = os.path.join(PROJECT_ROOT_DIR, SIMULATION_SCRIPT)
        print(f"[SERVER] Executing: {sys.executable} {script_abs_path}")
        process = await asyncio.create_subprocess_exec(
            sys.executable, # Use the same python interpreter running the server
            script_abs_path, # Use absolute path to script
            stdin=asyncio.subprocess.PIPE, # Allow sending commands via stdin
            stdout=asyncio.subprocess.PIPE, # Capture standard output
            stderr=asyncio.subprocess.PIPE, # Capture standard error
            cwd=PROJECT_ROOT_DIR, # Run script from the project root
            env=env # Pass the configured environment
        )
        print(f"[SERVER] Simulation process started with PID: {process.pid}")

        # Process stdout and stderr streams concurrently
        async def stream_output(stream, stream_name):
            while True:
                line_bytes = await stream.readline()
                if not line_bytes:
                    break # End of this stream
                line = line_bytes.decode('utf-8', errors='replace').strip()

                # Check if the line is a Base64 image data URI (from stdout)
                if stream_name == "STDOUT" and line.startswith("data:image"):
                    await broadcast_message(line) # Send image data to clients
                    await _persist_frame_from_data_url(line)
                else:
                    # Process other output (status, debug, errors)
                    log_prefix = f"[Sim {stream_name}]"
                    status_prefix = "status:running:"
                    # Filter out noisy plugin warnings
                    if "PluginManager" in line or "gym/spaces/box.py" in line:
                        print(f"{log_prefix} (info): {line}")
                        continue

                    normalized_line = line.lower()
                    is_error = False

                    # Treat genuine exceptions/tracebacks as errors, but allow informational stderr output
                    if "traceback" in normalized_line or "exception" in normalized_line or "critical" in normalized_line or "fatal" in normalized_line:
                        is_error = True

                    if "[error" in normalized_line and not is_error:
                        # Many controllable_sim messages use the word ERROR for emphasis; log as warning only
                        print(f"{log_prefix} WARN: {line}")
                        if "ready to receive commands" in normalized_line:
                            simulation_status["status"] = "running"
                            simulation_status["message"] = "Ready to receive commands."
                            await broadcast_message("status:running:Ready to receive commands.")
                        elif "robot is idle" in normalized_line:
                            simulation_status["status"] = "running"
                            simulation_status["message"] = "Robot is idle and awaiting commands."
                            await broadcast_message("status:running:Robot idle; awaiting commands.")
                        continue

                    if is_error:
                         print(f"{log_prefix} ERROR: {line}")
                         status_prefix = "status:error:" # Mark as error status
                         # Update global status immediately on error
                         simulation_status["status"] = "error"
                         simulation_status["message"] = line[:200] # Truncate long errors
                    elif "[DEBUG" in line or "[SCENE_MAPPING]" in line:
                         print(f"{log_prefix} DEBUG: {line}") # Just log debug lines
                         continue # Don't broadcast debug lines unless needed
                    else:
                         print(f"{log_prefix}: {line}") # Log other lines

                    if "ready to receive commands" in normalized_line:
                        simulation_status["status"] = "running"
                        simulation_status["message"] = "Ready to receive commands."
                        await broadcast_message("status:running:Ready to receive commands.")
                        continue
                    if "robot is idle" in normalized_line:
                        simulation_status["status"] = "running"
                        simulation_status["message"] = "Robot is idle and awaiting commands."
                        await broadcast_message("status:running:Robot idle; awaiting commands.")
                        continue

                    # Send status/error updates over WebSocket
                    await broadcast_message(f"{status_prefix}{line[:200]}") # Send truncated status

        # Store process reference globally
        simulation_process = process
        
        # Command forwarding task
        async def forward_commands():
            """Forward queued commands to the simulation process via stdin."""
            try:
                print(f"[ROBOT_CMD] Command forwarding task started. Process stdin: {process.stdin is not None}")
                while process.returncode is None:
                    try:
                        # Wait for a command with timeout
                        if robot_command_queue is not None:
                            try:
                                command = await asyncio.wait_for(robot_command_queue.get(), timeout=0.1)
                                if process.stdin is None:
                                    print(f"[ROBOT_CMD] ERROR: process.stdin is None, cannot send command: {command}")
                                    continue
                                
                                # Check if stdin is closed
                                if process.stdin.is_closing():
                                    print(f"[ROBOT_CMD] ERROR: stdin is closing, cannot send command: {command}")
                                    break
                                
                                command_bytes = f"{command}\n".encode('utf-8')
                                process.stdin.write(command_bytes)
                                await process.stdin.drain()
                                print(f"[ROBOT_CMD] Successfully sent command to simulation: {command}")
                            except asyncio.TimeoutError:
                                # No command available, continue waiting
                                continue
                            except BrokenPipeError as e:
                                print(f"[ROBOT_CMD] Broken pipe error (stdin closed): {e}")
                                break
                            except OSError as e:
                                print(f"[ROBOT_CMD] OS error writing to stdin: {e}")
                                break
                        else:
                            await asyncio.sleep(0.1)
                    except Exception as e:
                        print(f"[ROBOT_CMD] Error forwarding command: {e}")
                        import traceback
                        traceback.print_exc()
                        break
            except Exception as e:
                print(f"[ROBOT_CMD] Command forwarding task error: {e}")
                import traceback
                traceback.print_exc()
        
        # Start command forwarding
        command_forward_task = asyncio.create_task(forward_commands())
        
        # Run stream readers concurrently (but not image streaming yet)
        await asyncio.gather(
            stream_output(process.stdout, "STDOUT"),
            stream_output(process.stderr, "STDERR")
        )
        
        # Cancel command forwarding task
        command_forward_task.cancel()
        try:
            await command_forward_task
        except asyncio.CancelledError:
            pass

        # Wait for the process to fully exit and get the return code
        await process.wait()

        # Final status update based on return code
        if process.returncode == 0:
             # Check if status was already set to error by stderr processing
            if simulation_status['status'] != 'error':
                simulation_status = {"status": "complete", "message": "Simulation finished successfully."}
                print("Simulation finished successfully.")
                await broadcast_message("status:complete:Simulation finished.")
        else:
             # If return code is non-zero and status isn't already 'error'
            if simulation_status['status'] != 'error':
                 final_error_msg = f"Simulation script failed with exit code {process.returncode}."
                 simulation_status = {"status": "error", "message": final_error_msg}
                 print(final_error_msg)
                 await broadcast_message(f"status:error:{final_error_msg}")

    except Exception as e:
        # Catch errors during process creation or stream handling
        error_msg = f"An error occurred managing the simulation process: {e}"
        print(error_msg)
        # Ensure status is updated only if not already set to error
        if simulation_status['status'] != 'error':
            simulation_status = {"status": "error", "message": error_msg}
            await broadcast_message(f"status:error:{error_msg}") # Send final error status
    finally:
        # Clear global process reference
        simulation_process = None
        
        # Ensure process is terminated if something went wrong unexpectedly
        if process and process.returncode is None:
            print("Terminating simulation process...")
            try:
                process.terminate()
                await process.wait()
            except ProcessLookupError:
                pass # Process already finished


# --- Endpoint to Start Simulation (Must be async def) ---
@app.post("/run-simulation")
async def start_simulation_endpoint(): # <-- Needs to be async
    """Kicks off the simulation process in the background."""
    global simulation_status
    if simulation_status["status"] == "running":
        return JSONResponse({"message": "A simulation is already in progress."}, status_code=409)

    # Run the simulation process in the background using asyncio.create_task
    # This allows the endpoint to return immediately
    asyncio.create_task(run_simulation_process())

    return {"message": "Simulation process started in background."}


@app.post("/stop-simulation")
async def stop_simulation_endpoint():
    """Stops the currently running simulation if there is one."""
    global simulation_process, simulation_status, robot_command_queue

    if simulation_process is None or simulation_process.returncode is not None:
        simulation_status = {"status": "idle", "message": "No active simulation."}
        await broadcast_message("status:idle:No active simulation.")
        robot_command_queue = asyncio.Queue()
        return {"message": "No running simulation."}

    simulation_status = {"status": "stopping", "message": "Stopping simulation..."}
    await broadcast_message("status:stopping:Stopping simulation...")

    try:
        try:
            simulation_process.terminate()
        except ProcessLookupError:
            pass

        try:
            await asyncio.wait_for(simulation_process.wait(), timeout=10)
        except asyncio.TimeoutError:
            try:
                simulation_process.kill()
                await simulation_process.wait()
            except ProcessLookupError:
                pass

        simulation_process = None
        robot_command_queue = asyncio.Queue()
        simulation_status = {"status": "idle", "message": "Simulation stopped."}
        await broadcast_message("status:idle:Simulation stopped.")
        return {"message": "Simulation stopped."}
    except Exception as e:
        error_msg = f"Failed to stop simulation: {e}"
        print(error_msg)
        simulation_status = {"status": "error", "message": error_msg}
        await broadcast_message(f"status:error:{error_msg[:200]}")
        return JSONResponse({"error": error_msg}, status_code=500)


@app.get("/status")
async def get_status():
    """Returns the current status (less critical now with WebSockets)."""
    # Add debug info
    debug_info = {
        **simulation_status,
        "simulation_process_exists": simulation_process is not None,
        "simulation_process_pid": simulation_process.pid if simulation_process else None,
        "simulation_process_returncode": simulation_process.returncode if simulation_process else None,
        "queue_initialized": robot_command_queue is not None,
    }
    return JSONResponse(debug_info)

@app.post("/robot-command")
async def robot_command_endpoint(command: RobotCommand):
    """Receives robot control commands and queues them for the simulation."""
    payload, status = await _queue_robot_command(command.command, source="json-endpoint")
    return JSONResponse(payload, status_code=status)


async def _queue_robot_command(command_str: str, source: str = "api"):
    """Internal helper that enqueues a robot command after validating backend state."""
    global simulation_process, robot_command_queue

    print(f"[ROBOT_CMD] Command received from {source}: {command_str}")

    if robot_command_queue is None:
        print("[ROBOT_CMD] Rejecting command: queue not initialized")
        return {"error": "Command queue not initialized"}, 500

    if simulation_status["status"] != "running":
        print(f"[ROBOT_CMD] Rejecting command: status={simulation_status['status']}, message={simulation_status.get('message')}")
        return {"error": "Simulation is not running"}, 400

    if simulation_process is None:
        print("[ROBOT_CMD] Rejecting command: simulation_process is None")
        return {"error": "Simulation process is not active (None)"}, 400

    if simulation_process.returncode is not None:
        print(f"[ROBOT_CMD] Rejecting command: process exited with {simulation_process.returncode}")
        return {"error": f"Simulation process has exited with code {simulation_process.returncode}"}, 400

    # Check if stdin is available
    if simulation_process.stdin is None:
        print("[ROBOT_CMD] Rejecting command: process stdin missing")
        return {"error": "Simulation process stdin is not available"}, 500

    if simulation_process.stdin.is_closing():
        print("[ROBOT_CMD] Rejecting command: process stdin closing")
        return {"error": "Simulation process stdin is closing"}, 500

    try:
        # Queue the command to be sent to the simulation process
        await robot_command_queue.put(command_str)
        print(f"[ROBOT_CMD] Queued command: {command_str} (queue size will increase)")
        return {"message": f"Command '{command_str}' queued successfully"}, 200
    except Exception as e:
        print(f"[ROBOT_CMD] Error queueing command: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to queue command: {str(e)}"}, 500


@app.post("/move/{direction}")
async def move_robot(direction: str):
    """
    Shortcut endpoints for discrete robot motion commands (forward/back/left/right/stop).
    """
    normalized = direction.lower()
    command_map = {
        "forward": "forward",
        "left": "left",
        "right": "right",
        "back": "backward",
        "backward": "backward",
        "reverse": "backward",
        "stop": "stop",
    }

    if normalized not in command_map:
        return JSONResponse(
            {"error": f"Unsupported move direction '{direction}'"},
            status_code=400,
        )

    command_str = command_map[normalized]
    payload, status = await _queue_robot_command(command_str, source=f"/move/{direction}")
    return JSONResponse(payload, status_code=status)

@app.get("/latest-image")
async def get_latest_image():
    """Returns the latest image from the trajectory directory as base64."""
    try:
        search_dirs: list[Path] = []
        global live_frame_session_dir

        if live_frame_session_dir and live_frame_session_dir.exists():
            search_dirs.append(live_frame_session_dir)

        # Fall back to latest directory in base if current not set
        if not search_dirs:
            session_dirs = [p for p in LIVE_FRAMES_BASE_DIR.iterdir() if p.is_dir()]
            session_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            if session_dirs:
                search_dirs.append(session_dirs[0])

        if not search_dirs:
            return JSONResponse({"error": "No live frame directory found"}, status_code=404)

        latest_dir = search_dirs[0]
        image_candidates = sorted(
            [
                path
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp")
                for path in latest_dir.glob(ext)
            ]
        )

        if not image_candidates:
            return JSONResponse({"error": "No images found"}, status_code=404)

        latest_image = image_candidates[-1]
        img_data = latest_image.read_bytes()
        mime = "image/png"
        suffix = latest_image.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif suffix == ".webp":
            mime = "image/webp"

        b64_string = base64.b64encode(img_data).decode('utf-8')

        return JSONResponse({
            "image": f"data:{mime};base64,{b64_string}",
            "filename": latest_image.name,
            "total_images": len(image_candidates)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    global robot_command_queue
    robot_command_queue = asyncio.Queue()
    print("[SERVER] Initialized robot command queue")

# --- Static File Serving ---
# Mount last so it doesn't override your API endpoints
print(f"Serving web interface from: {WEB_DIR}")
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")