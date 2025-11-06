import os
import subprocess
import uuid
import asyncio # For handling subprocess streams
import sys # To find the python executable
import glob
import base64
from typing import Union

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
                line = line_bytes.decode('utf-8').strip()

                # Check if the line is a Base64 image data URI (from stdout)
                if stream_name == "STDOUT" and line.startswith("data:image/png;base64,"):
                    await broadcast_message(line) # Send image data to clients
                else:
                    # Process other output (status, debug, errors)
                    log_prefix = f"[Sim {stream_name}]"
                    status_prefix = "status:running:"
                    if "[ERROR" in line or "Traceback" in line or stream_name == "STDERR":
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
        
        # Start image streaming task
        trajectory_dir = os.path.join(PROJECT_ROOT_DIR, "data/trajectories")
        image_stream_task = asyncio.create_task(stream_trajectory_images(trajectory_dir))
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
        
        # Cancel image streaming task
        image_stream_task.cancel()
        try:
            await image_stream_task
        except asyncio.CancelledError:
            pass

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


@app.get("/status")
async def get_status():
    """Returns the current status (less critical now with WebSockets)."""
    return JSONResponse(simulation_status)

@app.post("/robot-command")
async def robot_command_endpoint(command: RobotCommand):
    """Receives robot control commands and queues them for the simulation."""
    global simulation_process, robot_command_queue
    
    if robot_command_queue is None:
        return JSONResponse(
            {"error": "Command queue not initialized"}, 
            status_code=500
        )
    
    if simulation_status["status"] != "running":
        return JSONResponse(
            {"error": "Simulation is not running"}, 
            status_code=400
        )
    
    if simulation_process is None:
        return JSONResponse(
            {"error": "Simulation process is not active (None)"}, 
            status_code=400
        )
    
    if simulation_process.returncode is not None:
        return JSONResponse(
            {"error": f"Simulation process has exited with code {simulation_process.returncode}"}, 
            status_code=400
        )
    
    # Check if stdin is available
    if simulation_process.stdin is None:
        return JSONResponse(
            {"error": "Simulation process stdin is not available"}, 
            status_code=500
        )
    
    if simulation_process.stdin.is_closing():
        return JSONResponse(
            {"error": "Simulation process stdin is closing"}, 
            status_code=500
        )
    
    try:
        # Queue the command to be sent to the simulation process
        await robot_command_queue.put(command.command)
        print(f"[ROBOT_CMD] Queued command: {command.command} (queue size will increase)")
        return JSONResponse({"message": f"Command '{command.command}' queued successfully"})
    except Exception as e:
        print(f"[ROBOT_CMD] Error queueing command: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"error": f"Failed to queue command: {str(e)}"}, 
            status_code=500
        )

@app.get("/latest-image")
async def get_latest_image():
    """Returns the latest image from the trajectory directory as base64."""
    try:
        trajectory_base_dir = os.path.join(PROJECT_ROOT_DIR, "data/trajectories")
        epidx_dirs = glob.glob(os.path.join(trajectory_base_dir, "epidx_*/main_agent/rgb"))
        
        if not epidx_dirs:
            return JSONResponse({"error": "No trajectory directory found"}, status_code=404)
        
        rgb_dir = max(epidx_dirs, key=os.path.getmtime)
        jpg_files = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
        
        if not jpg_files:
            return JSONResponse({"error": "No images found"}, status_code=404)
        
        # Get the latest image
        latest_image = jpg_files[-1]
        
        with open(latest_image, 'rb') as f:
            img_data = f.read()
        
        b64_string = base64.b64encode(img_data).decode('utf-8')
        
        return JSONResponse({
            "image": f"data:image/jpeg;base64,{b64_string}",
            "filename": os.path.basename(latest_image),
            "total_images": len(jpg_files)
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