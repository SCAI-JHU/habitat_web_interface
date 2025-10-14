import os
import subprocess
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Initialize the FastAPI app
app = FastAPI()

# --- Global State Management ---
# This simple dictionary will keep track of our simulation's status.
# In a real application, you might use a database or a more robust solution.
simulation_status = {
    "status": "idle",  # Can be 'idle', 'running', 'complete', or 'error'
    "video_url": None,
}

# --- Directory and Script Paths ---
# Get project root directory dynamically
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to project root
WEB_DIR = os.path.join(PROJECT_ROOT_DIR, "web")
OUTPUTS_DIR = os.path.join(WEB_DIR, "outputs")
SIMULATION_SCRIPT = "habitat_llm/examples/scene_mapping.py"
VIDEO_SCRIPT = "create_video.py"

# Ensure the output directory for videos exists
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ... (rest of your script) ...

# In server.py, replace the whole run_simulation function with this:

# In server.py, replace the whole run_simulation function with this:


@app.post("/run-simulation")
async def run_simulation():
    """
    This endpoint kicks off the simulation process in the background.
    """
    global simulation_status
    if simulation_status["status"] == "running":
        return JSONResponse(
            {"message": "A simulation is already in progress."}, status_code=409
        )

    print("Starting simulation...")
    simulation_status = {"status": "running", "video_url": None}

    video_filename = f"simulation_{uuid.uuid4().hex}.mp4"
    output_video_path = os.path.join(OUTPUTS_DIR, video_filename)

    try:
        print(f"Running habitat simulation script: {SIMULATION_SCRIPT}")
        # Set up environment with GPU device
        env = os.environ.copy()
        
        # Configure GPU devices for cluster environments
        # You can set these environment variables before starting the server:
        # export HABITAT_GPU_ID=0
        # export CUDA_VISIBLE_DEVICES=0
        # export EGL_DEVICE_ID=0
        
        habitat_gpu_id = os.environ.get("HABITAT_GPU_ID", "0")
        env["HABITAT_GPU_ID"] = habitat_gpu_id
        
        if "CUDA_VISIBLE_DEVICES" not in env:
            # If not set by SLURM, try to use the first GPU
            env["CUDA_VISIBLE_DEVICES"] = habitat_gpu_id
        
        # For headless rendering on clusters, EGL device must match
        if "EGL_DEVICE_ID" not in env:
            # Try to match EGL device to the CUDA device
            env["EGL_DEVICE_ID"] = habitat_gpu_id
        
        print(f"GPU Configuration:")
        print(f"  HABITAT_GPU_ID={env.get('HABITAT_GPU_ID')}")
        print(f"  CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}")
        print(f"  EGL_DEVICE_ID={env.get('EGL_DEVICE_ID')}")
        
        sim_process = subprocess.Popen(
            ["python", SIMULATION_SCRIPT], cwd=PROJECT_ROOT_DIR, env=env
        )
        sim_process.wait()

        if sim_process.returncode != 0:
            raise Exception("Simulation script failed.")

        # --- START OF NEW ROBUST PATH CODE ---
        # Use dynamic path for trajectories directory
        trajectories_base_dir = os.path.join(PROJECT_ROOT_DIR, "data/trajectories")

        all_traj_dirs = [
            os.path.join(trajectories_base_dir, d)
            for d in os.listdir(trajectories_base_dir)
            if os.path.isdir(os.path.join(trajectories_base_dir, d))
        ]

        if not all_traj_dirs:
            raise Exception("No trajectory directories found.")

        # Sort directories by modification time, newest first
        all_traj_dirs.sort(key=os.path.getmtime, reverse=True)

        image_frames_dir = None
        # Loop through the directories to find the newest one that is valid
        for traj_dir in all_traj_dirs:
            potential_path = os.path.join(traj_dir, "main_agent", "rgb")
            if os.path.exists(potential_path):
                image_frames_dir = potential_path
                break  # Found a valid one, stop looking

        if image_frames_dir is None:
            raise Exception("Could not find any valid trajectory with an rgb folder.")

        print(
            f"Found latest valid trajectory. Creating video from frames in: {image_frames_dir}"
        )
        # --- END OF NEW ROBUST PATH CODE ---

        video_process = subprocess.Popen(
            [
                "python",
                VIDEO_SCRIPT,
                "--image_dir",
                image_frames_dir,
                "--output_file",
                output_video_path,
            ],
            cwd=PROJECT_ROOT_DIR,
        )
        video_process.wait()

        if video_process.returncode != 0:
            raise Exception("Video creation script failed.")

        print(f"Process complete. Video available at: {output_video_path}")
        simulation_status = {
            "status": "complete",
            "video_url": f"/outputs/{video_filename}",
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        simulation_status = {"status": "error", "video_url": None}

    return {"message": "Simulation process started."}


@app.get("/status")
async def get_status():
    """
    This endpoint allows the frontend to poll for the simulation's status.
    """
    return JSONResponse(simulation_status)


# --- Static File Serving ---
# This will serve our index.html and the generated videos.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")
