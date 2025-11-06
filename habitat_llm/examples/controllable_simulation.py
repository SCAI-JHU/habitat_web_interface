#!/usr/bin/env python3
"""
Controllable simulation script that accepts commands via stdin for real-time robot control.
Based on interactive_play.py and scene_mapping.py
"""

import os
import sys
import select
import threading
import functools
print = functools.partial(print, flush=True)

# For image streaming
import base64
import io
import numpy as np
import imageio

# Append the path of the parent directory
sys.path.append("..")

from habitat_llm.utils import cprint, setup_config
from habitat_llm.agent.env import EnvironmentInterface, register_actions, register_sensors
from habitat_llm.evaluation import CentralizedEvaluationRunner
from habitat_llm.world_model import Room
from habitat_llm.utils.core import get_config
from habitat_llm.agent.env.dataset import CollaborationDatasetV0

def send_frame_to_stdout(frame_array):
    """Encodes a frame array (NumPy) into a Base64 string and prints it."""
    try:
        if frame_array.dtype != np.uint8:
            if frame_array.max() <= 1.0:
                frame_array = (frame_array * 255).astype(np.uint8)
            else:
                frame_array = frame_array.astype(np.uint8)
        
        if frame_array.shape[2] == 4:
            frame_array = frame_array[:, :, :3]

        buffer = io.BytesIO()
        imageio.imwrite(buffer, frame_array, format='PNG')
        b64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
        print(f"data:image/png;base64,{b64_string}", flush=True)
    except Exception as e:
        print(f"[ERROR_STREAM] Failed to encode/send frame: {e}", flush=True)

# Command queue for stdin commands
command_queue = []
command_lock = threading.Lock()

def read_stdin_commands():
    """Read commands from stdin in a separate thread."""
    global command_queue
    print("[STDIN_READER] Starting stdin reader thread...", file=sys.stderr, flush=True)
    while True:
        try:
            # Check if stdin is available (non-blocking)
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                line = sys.stdin.readline().strip()
                if line:
                    with command_lock:
                        command_queue.append(line)
                    print(f"[STDIN_READER] Received command: {line}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[STDIN_READER] Error reading stdin: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            break

def get_next_command():
    """Get the next command from the queue, if any."""
    with command_lock:
        if command_queue:
            return command_queue.pop(0)
    return None

def map_command_to_action(command: str, env_interface):
    """
    Map web control commands to robot actions.
    Returns low_level_action dict for agent 0, or None if no action.
    The format should match what the environment expects.
    """
    import numpy as np
    
    # Check what actions are available in the action space
    action_space = env_interface.action_space.spaces
    
    # Build the action dict - format depends on what's in action_space
    action_dict = {}
    
    if command == 'forward':
        # Forward movement
        if 'base_velocity' in action_space or 'base_velocity_non_cylinder' in action_space:
            action_key = 'base_velocity' if 'base_velocity' in action_space else 'base_velocity_non_cylinder'
            action_dict[action_key] = {'base_vel': np.array([1.0, 0.0], dtype=np.float32)}
    elif command == 'backward':
        if 'base_velocity' in action_space or 'base_velocity_non_cylinder' in action_space:
            action_key = 'base_velocity' if 'base_velocity' in action_space else 'base_velocity_non_cylinder'
            action_dict[action_key] = {'base_vel': np.array([-1.0, 0.0], dtype=np.float32)}
    elif command == 'left':
        if 'base_velocity' in action_space or 'base_velocity_non_cylinder' in action_space:
            action_key = 'base_velocity' if 'base_velocity' in action_space else 'base_velocity_non_cylinder'
            action_dict[action_key] = {'base_vel': np.array([0.0, 1.0], dtype=np.float32)}
    elif command == 'right':
        if 'base_velocity' in action_space or 'base_velocity_non_cylinder' in action_space:
            action_key = 'base_velocity' if 'base_velocity' in action_space else 'base_velocity_non_cylinder'
            action_dict[action_key] = {'base_vel': np.array([0.0, -1.0], dtype=np.float32)}
    elif command == 'arm_up':
        if 'arm_action' in action_space:
            # Arm joint 1 (lift) - need to check actual action space shape
            arm_space = action_space['arm_action'].spaces['arm_action']
            arm_action = np.zeros(arm_space.shape[0], dtype=np.float32)
            if len(arm_action) > 1:
                arm_action[1] = 0.1  # Small positive value for lift
            action_dict['arm_action'] = {'arm_action': arm_action}
    elif command == 'arm_down':
        if 'arm_action' in action_space:
            arm_space = action_space['arm_action'].spaces['arm_action']
            arm_action = np.zeros(arm_space.shape[0], dtype=np.float32)
            if len(arm_action) > 1:
                arm_action[1] = -0.1  # Small negative value for lower
            action_dict['arm_action'] = {'arm_action': arm_action}
    elif command == 'grip_open':
        if 'arm_action' in action_space:
            arm_space = action_space['arm_action'].spaces['arm_action']
            arm_action = np.zeros(arm_space.shape[0], dtype=np.float32)
            if len(arm_action) > 0:
                arm_action[-1] = -1.0  # Last element for grip
            action_dict['arm_action'] = {'arm_action': arm_action}
    elif command == 'grip_close':
        if 'arm_action' in action_space:
            arm_space = action_space['arm_action'].spaces['arm_action']
            arm_action = np.zeros(arm_space.shape[0], dtype=np.float32)
            if len(arm_action) > 0:
                arm_action[-1] = 1.0  # Last element for grip
            action_dict['arm_action'] = {'arm_action': arm_action}
    elif command == 'stop':
        if 'base_velocity' in action_space or 'base_velocity_non_cylinder' in action_space:
            action_key = 'base_velocity' if 'base_velocity' in action_space else 'base_velocity_non_cylinder'
            action_dict[action_key] = {'base_vel': np.array([0.0, 0.0], dtype=np.float32)}
    else:
        return None
    
    if action_dict:
        return {0: action_dict}
    return None

def run_controllable_simulation():
    """Run simulation with real-time control via stdin commands."""
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    
    seed = 47668090
    gpu_device_id = os.environ.get("HABITAT_GPU_ID", "0")
    print(f"Using GPU device ID: {gpu_device_id}")
    
    robot_urdf_path = os.path.join(project_root, "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf")
    
    DATASET_OVERRIDES = [
        "habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz",
        "habitat.dataset.scenes_dir=data/hssd-hab/",
        f"habitat.simulator.agents.main_agent.articulated_agent_urdf={robot_urdf_path}",
        "habitat.simulator.agents.main_agent.articulated_agent_type=SpotRobot",
        f"habitat.simulator.habitat_sim_v0.gpu_device_id={gpu_device_id}",
    ]
    
    SENSOR_OVERRIDES = [
        "habitat.simulator.agents.main_agent.sim_sensors.jaw_depth_sensor.normalize_depth=False"
    ]
    
    LLM_OVERRIDES = [
        "llm@evaluation.planner.plan_config.llm=mock",
    ]
    
    TRAJECTORY_OVERRIDES = [
        "evaluation.save_video=True",
        "evaluation.output_dir=./outputs",
        "trajectory.save=True",
        "trajectory.agent_names=[main_agent]",
    ]
    
    EPISODE_OVERRIDES = ["+episode_indices=[2]"]
    
    config_base = get_config(
        "examples/single_agent_scene_mapping.yaml",
        overrides=DATASET_OVERRIDES
        + SENSOR_OVERRIDES
        + LLM_OVERRIDES
        + TRAJECTORY_OVERRIDES
        + EPISODE_OVERRIDES,
    )
    config = setup_config(config_base, seed)
    
    if config == None:
        cprint("Failed to setup config. Exiting", "red")
        return
    
    register_sensors(config)
    register_actions(config)
    
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    if config.get("episode_indices", None) is not None:
        episode_subset = [dataset.episodes[x] for x in config.episode_indices]
        dataset = CollaborationDatasetV0(
            config=config.habitat.dataset, episodes=episode_subset
        )
    env_interface = EnvironmentInterface(config, dataset=dataset)
    
    # Start stdin reading thread
    print("[CONTROLLABLE_SIM] Starting stdin command reader thread...", file=sys.stderr, flush=True)
    stdin_thread = threading.Thread(target=read_stdin_commands, daemon=True)
    stdin_thread.start()
    print("[CONTROLLABLE_SIM] Stdin reader thread started", file=sys.stderr, flush=True)
    
    print("[CONTROLLABLE_SIM] =========================================", file=sys.stderr, flush=True)
    print("[CONTROLLABLE_SIM] STARTING CONTROLLABLE SIMULATION", file=sys.stderr, flush=True)
    print("[CONTROLLABLE_SIM] This script WAITS for commands - NO autonomous behavior", file=sys.stderr, flush=True)
    print("[CONTROLLABLE_SIM] =========================================", file=sys.stderr, flush=True)
    print("[CONTROLLABLE_SIM] Starting controllable simulation...", file=sys.stderr, flush=True)
    print("[CONTROLLABLE_SIM] Ready to receive commands via stdin", file=sys.stderr, flush=True)
    
    # Reset environment
    print("[CONTROLLABLE_SIM] Resetting environment...", file=sys.stderr, flush=True)
    env_interface.reset_environment()
    raw_obs = env_interface.get_observations()
    observations = env_interface.parse_observations(raw_obs)
    
    # Send initial frame
    if "head_rgb" in raw_obs:
        frame = raw_obs["head_rgb"]
        if hasattr(frame, 'cpu'):
            frame = frame.cpu().numpy()
        send_frame_to_stdout(frame)
    
    step_count = 0
    max_steps = 10000
    
    print("[CONTROLLABLE_SIM] =========================================", file=sys.stderr, flush=True)
    print("[CONTROLLABLE_SIM] Entering control loop - waiting for commands...", file=sys.stderr, flush=True)
    print("[CONTROLLABLE_SIM] Robot is IDLE - will NOT move until you send a command", file=sys.stderr, flush=True)
    print("[CONTROLLABLE_SIM] Send commands via web interface buttons", file=sys.stderr, flush=True)
    print("[CONTROLLABLE_SIM] =========================================", file=sys.stderr, flush=True)
    
    # Main control loop - ONLY steps when commands are received
    print("[CONTROLLABLE_SIM] =========================================", file=sys.stderr, flush=True)
    print("[CONTROLLABLE_SIM] CONTROL LOOP STARTED - Robot will NOT move autonomously", file=sys.stderr, flush=True)
    print("[CONTROLLABLE_SIM] Waiting for first command...", file=sys.stderr, flush=True)
    print("[CONTROLLABLE_SIM] =========================================", file=sys.stderr, flush=True)
    
    while step_count < max_steps:
        # Block until we get a command - NO AUTONOMOUS BEHAVIOR
        command = None
        wait_count = 0
        while command is None:
            command = get_next_command()
            if command is None:
                # No command yet - wait a bit and check again
                wait_count += 1
                if wait_count % 100 == 0:  # Log every 10 seconds (100 * 0.1s)
                    print(f"[CONTROLLABLE_SIM] Still waiting for commands... (waited {wait_count * 0.1:.1f}s)", file=sys.stderr, flush=True)
                import time
                time.sleep(0.1)  # 100ms polling
        
        # We have a command!
        print(f"[CONTROLLABLE_SIM] =========================================", file=sys.stderr, flush=True)
        print(f"[CONTROLLABLE_SIM] Received command: {command}", file=sys.stderr, flush=True)
        
        low_level_action = map_command_to_action(command, env_interface)
        if low_level_action:
            # Apply the action - this will step the simulation
            obs, reward, done, info = env_interface.step(low_level_action)
            observations = env_interface.parse_observations(obs)
            step_count += 1
            
            # Send updated frame
            if "head_rgb" in obs:
                frame = obs["head_rgb"]
                if hasattr(frame, 'cpu'):
                    frame = frame.cpu().numpy()
                send_frame_to_stdout(frame)
            
            print(f"[CONTROLLABLE_SIM] Executed command: {command} (step {step_count})", file=sys.stderr, flush=True)
        else:
            print(f"[CONTROLLABLE_SIM] Unknown command: {command}", file=sys.stderr, flush=True)
    
    print("[CONTROLLABLE_SIM] Simulation completed", file=sys.stderr, flush=True)
    env_interface.sim.close()

if __name__ == "__main__":
    cprint("\nStarting controllable simulation", "blue")
    run_controllable_simulation()
    cprint("\nEnd of controllable simulation", "blue")

