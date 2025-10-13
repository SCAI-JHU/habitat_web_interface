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
# Make sure these paths are correct for your system.
WEB_DIR = "web"
OUTPUTS_DIR = os.path.join(WEB_DIR, "outputs")
SIMULATION_SCRIPT = "habitat_llm/examples/scene_mapping.py"
VIDEO_SCRIPT = "create_video.py"

# This is the directory where the simulation saves its raw image frames.
# We'll need to find a way to make this dynamic later, but for now, we'll use the one from your script.
# IMPORTANT: Update this path to be correct.
IMAGE_FRAMES_DIR = "/home/oakers1/scratchtshu2/oakers1/partnr-planner/data/trajectories/epidx_0_scene_106366386_174226770/main_agent/rgb"

# Ensure the output directory for videos exists
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# --- API Endpoints ---
# --- Add this new variable near the top of your script ---
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

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
        sim_process = subprocess.Popen(
            ["python", SIMULATION_SCRIPT], cwd=PROJECT_ROOT_DIR
        )
        sim_process.wait()

        if sim_process.returncode != 0:
            raise Exception("Simulation script failed.")

        # --- START OF NEW ROBUST PATH CODE ---
        trajectories_base_dir = ( "/weka/scratch/tshu2/oakers1/partnr-planner/data/trajectories"
        )

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
