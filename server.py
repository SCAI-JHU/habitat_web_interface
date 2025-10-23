import os
import subprocess
import uuid
import asyncio # For handling subprocess streams
import sys # To find the python executable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect # Import WebSocket components
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Initialize the FastAPI app
app = FastAPI()

# --- Global State Management ---
simulation_status = {
    "status": "idle",
    "message": "Ready to start",
}
# Keep track of connected WebSocket clients
connected_clients: list[WebSocket] = []


# --- Directory and Script Paths ---
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(PROJECT_ROOT_DIR, "web")
SIMULATION_SCRIPT = "habitat_llm/examples/scene_mapping.py"

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

    print(f"Running habitat simulation script: {SIMULATION_SCRIPT}")
    print(f"GPU Configuration: CUDA={env.get('CUDA_VISIBLE_DEVICES')}, EGL={env.get('EGL_DEVICE_ID')}")

    process = None # Define process variable outside try block
    try:
        # Use asyncio's subprocess handling for non-blocking I/O
        process = await asyncio.create_subprocess_exec(
            sys.executable, # Use the same python interpreter running the server
            SIMULATION_SCRIPT,
            stdout=asyncio.subprocess.PIPE, # Capture standard output
            stderr=asyncio.subprocess.PIPE, # Capture standard error
            cwd=PROJECT_ROOT_DIR, # Run script from the project root
            env=env # Pass the configured environment
        )

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

        # Run stream readers concurrently
        await asyncio.gather(
            stream_output(process.stdout, "STDOUT"),
            stream_output(process.stderr, "STDERR")
        )

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


# --- Static File Serving ---
# Mount last so it doesn't override your API endpoints
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")