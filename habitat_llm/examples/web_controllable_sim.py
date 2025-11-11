#!/usr/bin/env python3
"""
Controllable simulation script FOR WEB.
Accepts commands via stdin for real-time robot control.

Combines the robust config loading from 'interactive_play_stretch.py'
with the stdin/stdout communication from 'controllable_simulation.py'.
"""

import os
import sys
import select
import threading
import functools
import argparse
import time
print = functools.partial(print, flush=True)

# For image streaming
import base64
import io
import numpy as np
import imageio
import torch

try:
    import gym
    from gym import spaces as gym_spaces
except ImportError:
    import gymnasium as gym
    from gymnasium import spaces as gym_spaces

# Append the path of the parent directory
sys.path.append("..")

import habitat
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    ThirdRGBSensorConfig,
    TopDownMapMeasurementConfig
)
from habitat.core.logging import logger
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay

from habitat_llm.utils import cprint, setup_config
from habitat_llm.agent.env import EnvironmentInterface, register_actions, register_sensors
from habitat_llm.evaluation import CentralizedEvaluationRunner
from habitat_llm.world_model import Room
from habitat_llm.utils.core import get_config
from habitat_llm.agent.env.dataset import CollaborationDatasetV0

# --- DEFAULT CONFIG PATH (from interactive_play_stretch.py) ---
DEFAULT_CFG = "benchmark/rearrange/play/play_stretch.yaml"

def send_frame_to_stdout(frame_array):
    """Encodes a frame array (NumPy) into a Base64 string and prints it."""
    try:
        frame_array = np.array(frame_array)

        if frame_array.dtype != np.uint8:
            if frame_array.max() <= 1.0:
                frame_array = (frame_array * 255).astype(np.uint8)
            else:
                frame_array = frame_array.astype(np.uint8)

        if frame_array.ndim == 4 and frame_array.shape[0] == 1:
            frame_array = frame_array[0]

        if frame_array.ndim == 3:
            if frame_array.shape[0] in (3, 4) and frame_array.shape[2] not in (3, 4):
                frame_array = np.transpose(frame_array, (1, 2, 0))

        if frame_array.ndim != 3 or frame_array.shape[2] not in (3, 4):
            print(f"[FRAME_WARN] Unexpected frame shape {frame_array.shape}, skipping frame", flush=True)
            return

        if frame_array.shape[2] == 1:
            frame_array = np.repeat(frame_array, 3, axis=2)
        elif frame_array.shape[2] == 4:
            frame_array = frame_array[:, :, :3]

        buffer = io.BytesIO()
        # --- CRITICAL FIX: Use JPEG and lower quality for smaller message size ---
        imageio.imwrite(buffer, frame_array, format='JPEG', quality=75)
        b64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
        print(f"data:image/jpeg;base64,{b64_string}", flush=True)
        # -----------------------------------------------------------------
        
    except Exception as e:
        print(f"[FRAME_WARN] Failed to encode/send frame: {e}", flush=True)

# Live frame capture helper
def send_current_rgb_frame(env_interface, obs_dict=None, label=""):
    """
    Grab the latest RGB sensor frame and push it through stdout.
    """
    try:
        if obs_dict is None:
            # We must use get_sensor_observations() for *all* agents
            obs_dict = env_interface.sim.get_sensor_observations() 
    except Exception as e:
        print(f"[FRAME] Failed to get sensor observations: {e}", flush=True)
        return

    if not obs_dict:
        print("[FRAME] No observations returned from simulator", flush=True)
        return

    # Get sensor keys for the *first* agent
    agent_mgr = env_interface.sim.agents_mgr
    if not agent_mgr:
        print("[FRAME] No agents found", flush=True)
        return

    # Use the keys from the *first agent* (agent 0)
    agent_obs_keys = list(agent_mgr[0].sensor_suite.sensors.keys())
    print(f"[DEBUG] Agent 0 sensor keys: {agent_obs_keys}", flush=True)

    # Prioritize sensors based on common names for the *Stretch* robot
    possible_rgb_keys = [
        "head_rgb",                   # Stretch's head camera
        "articulated_agent_jaw_rgb",  # Spot's camera (fallback)
        "third_rgb",                  # Third person (fallback)
        "rgb",
    ]
    
    rgb_frame = None
    
    # Try to find a sensor from our priority list in the *full* obs_dict
    for key in possible_rgb_keys:
        if key in obs_dict:
            rgb_frame = obs_dict[key]
            print(f"[DEBUG] Found frame using key: {key}", flush=True)
            break

    if rgb_frame is None:
        print("[FRAME] No high-priority RGB frame found in observations", flush=True)
        return

    if hasattr(rgb_frame, "cpu"):
        rgb_frame = rgb_frame.cpu().numpy()

    send_frame_to_stdout(rgb_frame)
    if label:
        print(f"[FRAME] Sent frame ({label})", flush=True)

# Command queue for stdin commands
command_queue = []
command_lock = threading.Lock()

# Action key globals
BASE_ACTION_KEY = None
BASE_ACTION_SUBKEY = None
ARM_ACTION_KEY = None
ARM_ACTION_SUBKEY = None
ROOT_ACTION_SPACE = None
HAS_MAGIC_GRASP = False

def read_stdin_commands():
    """Read commands from stdin in a separate thread."""
    global command_queue
    print("[STDIN_READER] Starting stdin reader thread...", file=sys.stderr, flush=True)
    while True:
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                line = sys.stdin.readline().strip()
                if line:
                    with command_lock:
                        command_queue.append(line)
                    print(f"[STDIN_READER] Received command: {line}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[STDIN_READER] Error reading stdin: {e}", file=sys.stderr, flush=True)
            break

def get_next_command():
    """Get the next command from the queue, if any."""
    with command_lock:
        if command_queue:
            return command_queue.pop(0)
    return None

def _ensure_action_keys(env_interface, task_config):
    """
    Inspect the environment action space once and cache the keys/subkeys.
    This version is adapted from controllable_simulation.py
    """
    global BASE_ACTION_KEY, ARM_ACTION_KEY, ROOT_ACTION_SPACE, HAS_MAGIC_GRASP

    if ROOT_ACTION_SPACE is not None or BASE_ACTION_KEY is not None:
        return # Already initialized

    action_space = env_interface.action_space
    print(f"[CONTROLLABLE_SIM] Action space type: {type(action_space)}", file=sys.stderr, flush=True)

    if isinstance(action_space, gym_spaces.Dict):
        keys = list(action_space.spaces.keys())
        print(f"[CONTROLLABLE_SIM] Action space keys: {keys}", file=sys.stderr, flush=True)

        for key in keys:
            if "base_velocity" in key.lower():
                BASE_ACTION_KEY = key
                print(f"[CONTROLLABLE_SIM] >>> Detected base action key: {BASE_ACTION_KEY}", file=sys.stderr, flush=True)
            if "arm_action" in key.lower():
                ARM_ACTION_KEY = key
                print(f"[CONTROLLABLE_SIM] >>> Detected arm action key: {ARM_ACTION_KEY}", file=sys.stderr, flush=True)
    
    if task_config.actions.get(ARM_ACTION_KEY) and "grip_controller" in task_config.actions[ARM_ACTION_KEY]:
        if "MagicGrasp" in task_config.actions[ARM_ACTION_KEY].grip_controller:
            HAS_MAGIC_GRASP = True
            print(f"[CONTROLLABLE_SIM] >>> Detected MagicGrasp controller", file=sys.stderr, flush=True)

    if BASE_ACTION_KEY is None and ARM_ACTION_KEY is None:
        if isinstance(action_space, gym_spaces.Box):
            ROOT_ACTION_SPACE = action_space
            print(f"[CONTROLLABLE_SIM] >>> Detected root Box action space: shape={action_space.shape}", file=sys.stderr, flush=True)
        else:
            print("[CONTROLLABLE_SIM] WARNING: Unsupported action space", type(action_space), file=sys.stderr, flush=True)

def map_command_to_action(command: str, env_interface):
    """
    Map web control commands to robot actions.
    This version is adapted for the Stretch robot's action space.
    """
    actions = {}
    
    # Base velocity (I, J, K, L from interactive_play_stretch.py)
    # Mapping web commands to these keys
    base_action = [0.0, 0.0]
    if command == 'forward':
        base_action = [1.0, 0.0]
    elif command == 'backward':
        base_action = [-1.0, 0.0]
    elif command == 'left':
        base_action = [0.0, 1.0] # 'J' key
    elif command == 'right':
        base_action = [0.0, -1.0] # 'L' key
    
    if BASE_ACTION_KEY:
        actions[BASE_ACTION_KEY] = { "base_vel": base_action }

    # Arm and Gripper
    if ARM_ACTION_KEY:
        # Get the arm action space shape
        arm_space_shape = env_interface.action_space.spaces[ARM_ACTION_KEY].spaces["arm_action"].shape
        arm_action = np.zeros(arm_space_shape, dtype=np.float32)
        grip_action = 0.0 # Default: do nothing

        if command == 'arm_up':
            arm_action[0] = 1.0 # Corresponds to 'Q' key (joint 0 up)
        elif command == 'arm_down':
            arm_action[0] = -1.0 # Corresponds to '1' key (joint 0 down)
        elif command == 'grip_open':
            grip_action = -1.0 # Desnap / Open
        elif command == 'grip_close':
            grip_action = 1.0 # Snap / Close
        
        actions[ARM_ACTION_KEY] = {
            "arm_action": arm_action,
            "grip_action": grip_action
        }

    if not actions:
        print(f"[CONTROLLABLE_SIM] Unknown command: {command}", file=sys.stderr, flush=True)
        return None

    return {0: actions} # Wrap for agent 0

def run_controllable_simulation(args):
    """Run simulation with real-time control via stdin commands."""
    
    # --- Config loading from interactive_play_stretch.py ---
    config = habitat.get_config(args.cfg, args.opts)
    with habitat.config.read_write(config):
        env_config = config.habitat.environment
        sim_config = config.habitat.simulator
        task_config = config.habitat.task

        # Add sensors required for web visualization
        agent_config = get_agent_config(sim_config=sim_config)
        agent_config.sim_sensors.update(
            {
                "third_rgb_sensor": ThirdRGBSensorConfig(
                    height=args.play_cam_res, width=args.play_cam_res
                )
            }
        )
        # Ensure top-down map is available if you want to add it later
        if "top_down_map" not in config.habitat.task.measurements:
            config.habitat.task.measurements.update(
                {"top_down_map": TopDownMapMeasurementConfig()}
            )

        if args.never_end:
            env_config.max_episode_steps = 0
            
        # Ensure we're using the Stretch robot's IK settings
        if not args.disable_inverse_kinematics:
            if "arm_action" not in task_config.actions:
                raise ValueError("Action space does not have arm control")
            sim_config.agents.main_agent.ik_arm_urdf = (
                "./data/robots/hab_stretch/urdf/hab_stretch.urdf"
            )
            task_config.actions.arm_action.arm_controller = (
                "ArmRelPosKinematicReducedActionStretch"
            )
            task_config.actions.arm_action.grip_controller = "MagicGraspAction"
    # --------------------------------------------------------

    # Initialize environment
    cprint("Loading environment with config...", "yellow")
    try:
        with habitat.Env(config=config) as env:
            env_interface = env # Use the fully initialized Env
            cprint("Environment loaded successfully.", "green")
            
            # Identify action keys
            _ensure_action_keys(env_interface, task_config)
            
            # Start stdin reading thread
            print("[CONTROLLABLE_SIM] Starting stdin command reader thread...", file=sys.stderr, flush=True)
            stdin_thread = threading.Thread(target=read_stdin_commands, daemon=True)
            stdin_thread.start()
            
            print("[CONTROLLABLE_SIM] =========================================", file=sys.stderr, flush=True)
            print("[CONTROLLABLE_SIM] STARTING WEB-CONTROLLABLE SIMULATION", file=sys.stderr, flush=True)
            print("[CONTROLLABLE_SIM] This script WAITS for commands - NO autonomous behavior", file=sys.stderr, flush=True)
            print("[CONTROLLABLE_SIM] =========================================", file=sys.stderr, flush=True)
            
            # Reset environment
            print("[CONTROLLABLE_SIM] Resetting environment...", file=sys.stderr, flush=True)
            raw_obs = env.reset()
            
            # Send initial frame immediately
            send_current_rgb_frame(env_interface, obs_dict=raw_obs, label="initial")
            
            step_count = 0
            
            print("[CONTROLLABLE_SIM] =========================================", file=sys.stderr, flush=True)
            print("[CONTROLLABLE_SIM] Entering control loop - waiting for commands...", file=sys.stderr, flush=True)
            print("[CONTROLLABLE_SIM] Robot is IDLE - will NOT move until you send a command", file=sys.stderr, flush=True)
            print("[CONTROLLABLE_SIM] =========================================", file=sys.stderr, flush=True)
            
            while True:
                command = get_next_command()
                
                if command is None:
                    # No command yet - wait and send idle frame
                    time.sleep(0.1) # 100ms polling
                    if step_count % 10 == 0: # Send idle frame every 1s
                        send_current_rgb_frame(env_interface, label="idle")
                    step_count += 1
                    continue
                
                # We have a command!
                print(f"[CONTROLLABLE_SIM] Received command: {command}", file=sys.stderr, flush=True)
                
                low_level_action = map_command_to_action(command, env_interface)
                
                if low_level_action:
                    # Apply the action
                    obs, reward, done, info = env_interface.step(low_level_action)
                    step_count += 1
                    
                    # Send updated frame
                    send_current_rgb_frame(env_interface, obs_dict=obs, label=f"after {command}")
                    
                    print(f"[CONTROLLABLE_SIM] Executed command: {command} (step {step_count})", file=sys.stderr, flush=True)

                    if done:
                        print("[CONTROLLABLE_SIM] Episode finished. Resetting...", file=sys.stderr, flush=True)
                        raw_obs = env.reset()
                        send_current_rgb_frame(env_interface, obs_dict=raw_obs, label="reset")

                else:
                    print(f"[CONTROLLABLE_SIM] Unknown command: {command}", file=sys.stderr, flush=True)
        
    except Exception as e:
        print(f"[CONTROLLABLE_SIM] CRITICAL ERROR: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
    finally:
        print("[CONTROLLABLE_SIM] Simulation ended.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    # --- Arg parsing from interactive_play_stretch.py ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--play-cam-res", type=int, default=512)
    parser.add_argument("--never-end", action="store_true", default=False)
    parser.add_argument("--disable-inverse-kinematics", action="store_true")
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    
    cprint("\nStarting web-controllable simulation", "blue")
    run_controllable_simulation(args)
    cprint("\nEnd of web-controllable simulation", "blue")