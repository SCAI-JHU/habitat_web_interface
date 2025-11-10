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
import time
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

from habitat_llm.utils import cprint, setup_config
from habitat_llm.agent.env import EnvironmentInterface, register_actions, register_sensors
from habitat_llm.evaluation import CentralizedEvaluationRunner
from habitat_llm.world_model import Room
from habitat_llm.utils.core import get_config
from habitat_llm.agent.env.dataset import CollaborationDatasetV0

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

        # Handle tensors that are channel-first
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
        imageio.imwrite(buffer, frame_array, format='PNG')
        b64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
        print(f"data:image/png;base64,{b64_string}", flush=True)
    except Exception as e:
        print(f"[FRAME_WARN] Failed to encode/send frame: {e}", flush=True)

# Live frame capture helper --------------------------------------------------
def send_current_rgb_frame(env_interface, obs_dict=None, label=""):
    """
    Grab the latest RGB sensor frame directly from the simulator (or obs dict)
    and push it through stdout so the server streams it to the frontend.
    """
    try:
        if obs_dict is None:
            obs_dict = env_interface.sim.get_sensor_observations()
    except Exception as e:
        print(f"[FRAME] Failed to get sensor observations: {e}", flush=True)
        return

    if not obs_dict:
        print("[FRAME] No observations returned from simulator", flush=True)
        return

    print(f"[DEBUG] Available sensor keys: {list(obs_dict.keys())}", flush=True)

    # Try common RGB sensor keys
    possible_rgb_keys = [
        "articulated_agent_jaw_rgb",  # <-- Try this first (often main nav camera for Spot)
        "head_rgb",                   # <-- Second best option
        "articulated_agent_arm_rgb",  # <-- Good if you are moving the arm
        "third_rgb",                  # <-- Likely static, keep last as fallback
        "rgb",
    ]
    
    rgb_frame = None
    for key in possible_rgb_keys:
        if key in obs_dict:
            rgb_frame = obs_dict[key]
            break

    if rgb_frame is None:
        # Fallback: pick first array that looks like an RGB image
        for key, value in obs_dict.items():
            if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[2] in (3, 4):
                rgb_frame = value
                break

    if rgb_frame is None:
        print("[FRAME] No RGB frame found in observations", flush=True)
        return

    if hasattr(rgb_frame, "cpu"):
        rgb_frame = rgb_frame.cpu().numpy()

    send_frame_to_stdout(rgb_frame)
    if label:
        print(f"[FRAME] Sent frame ({label})", flush=True)

# Command queue for stdin commands
command_queue = []
command_lock = threading.Lock()

# Navigation skill support
nav_skill = None
NAV_FORWARD_DISTANCE = 0.35
NAV_TURN_DISTANCE = 0.15
NAV_TURN_ANGLE_DEG = 25.0
NAV_MAX_STEPS_PER_COMMAND = 12

# Action key globals - populated after environment initialization
BASE_ACTION_KEY = None
BASE_ACTION_SUBKEY = None
ARM_ACTION_KEY = None
ARM_ACTION_SUBKEY = None
ROOT_ACTION_SPACE = None

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

def _ensure_action_keys(env_interface):
    """
    Inspect the environment action space once and cache the keys/subkeys
    for base/arm control so we can construct actions correctly.
    """
    global BASE_ACTION_KEY, BASE_ACTION_SUBKEY, ARM_ACTION_KEY, ARM_ACTION_SUBKEY, ROOT_ACTION_SPACE
    if (BASE_ACTION_KEY is not None or ARM_ACTION_KEY is not None or ROOT_ACTION_SPACE is not None):
        return

    action_space = env_interface.action_space
    print(f"[CONTROLLABLE_SIM] Action space type: {type(action_space)}", file=sys.stderr, flush=True)
    if isinstance(action_space, gym_spaces.Dict):
        keys = list(action_space.spaces.keys())
        print(f"[CONTROLLABLE_SIM] Action space keys: {keys}", file=sys.stderr, flush=True)

        for key in keys:
            space = action_space.spaces[key]
            lowered = key.lower()
            print(f"[CONTROLLABLE_SIM]   - {key}: {space}", file=sys.stderr, flush=True)
            if BASE_ACTION_KEY is None and "base_velocity" in lowered:
                BASE_ACTION_KEY = key
                if isinstance(space, gym_spaces.Dict):
                    BASE_ACTION_SUBKEY = next(iter(space.spaces.keys()))
                else:
                    BASE_ACTION_SUBKEY = None
                print(f"[CONTROLLABLE_SIM] >>> Detected base action key: {BASE_ACTION_KEY} (subkey={BASE_ACTION_SUBKEY})", file=sys.stderr, flush=True)
            if ARM_ACTION_KEY is None and "arm" in lowered:
                ARM_ACTION_KEY = key
                if isinstance(space, gym_spaces.Dict):
                    ARM_ACTION_SUBKEY = next(iter(space.spaces.keys()))
                else:
                    ARM_ACTION_SUBKEY = None
                print(f"[CONTROLLABLE_SIM] >>> Detected arm action key: {ARM_ACTION_KEY} (subkey={ARM_ACTION_SUBKEY})", file=sys.stderr, flush=True)
    else:
        if isinstance(action_space, gym_spaces.Box):
            ROOT_ACTION_SPACE = action_space
            print(f"[CONTROLLABLE_SIM] >>> Detected root Box action space: shape={action_space.shape}", file=sys.stderr, flush=True)
        else:
            print("[CONTROLLABLE_SIM] WARNING: Unsupported action space type", type(action_space), file=sys.stderr, flush=True)


def _make_base_action(env_interface, forward: float, turn: float):
    global BASE_ACTION_KEY, BASE_ACTION_SUBKEY
    if BASE_ACTION_KEY is None:
        return None
    base_space = env_interface.action_space.spaces[BASE_ACTION_KEY]
    if isinstance(base_space, gym_spaces.Dict):
        sub_space = base_space.spaces[BASE_ACTION_SUBKEY]
        vec = np.zeros(sub_space.shape, dtype=np.float32)
        if vec.shape[0] > 0:
            vec[0] = forward
        if vec.shape[0] > 1:
            vec[1] = turn
        return {BASE_ACTION_KEY: {BASE_ACTION_SUBKEY: vec}}
    elif isinstance(base_space, gym_spaces.Box):
        vec = np.zeros(base_space.shape, dtype=np.float32)
        if base_space.shape[0] > 0:
            vec[0] = forward
        if base_space.shape[0] > 1:
            vec[1] = turn
        return {BASE_ACTION_KEY: vec}
    else:
        print(f"[CONTROLLABLE_SIM] WARNING: Unsupported base action space type {type(base_space)}", file=sys.stderr, flush=True)
        return None


def _make_root_box_action(forward: float, turn: float):
    global ROOT_ACTION_SPACE
    if ROOT_ACTION_SPACE is None:
        return None
    vec = np.zeros(ROOT_ACTION_SPACE.shape, dtype=np.float32)
    if vec.shape[0] > 0:
        vec[0] = forward
    if vec.shape[0] > 1:
        vec[1] = turn
    return {0: vec}


def _make_arm_action(env_interface, joint_index: int, amount: float):
    global ARM_ACTION_KEY, ARM_ACTION_SUBKEY
    if ARM_ACTION_KEY is None:
        return None
    def _resolve_index(length):
        if joint_index == -1:
            return length - 1
        return joint_index
    arm_space = env_interface.action_space.spaces[ARM_ACTION_KEY]
    if isinstance(arm_space, gym_spaces.Dict):
        sub_space = arm_space.spaces[ARM_ACTION_SUBKEY]
        vec = np.zeros(sub_space.shape, dtype=np.float32)
        idx = _resolve_index(vec.shape[0])
        if 0 <= idx < vec.shape[0]:
            vec[idx] = amount
        return {ARM_ACTION_KEY: {ARM_ACTION_SUBKEY: vec}}
    elif isinstance(arm_space, gym_spaces.Box):
        vec = np.zeros(arm_space.shape, dtype=np.float32)
        idx = _resolve_index(vec.shape[0])
        if 0 <= idx < vec.shape[0]:
            vec[idx] = amount
        return {ARM_ACTION_KEY: vec}
    else:
        print(f"[CONTROLLABLE_SIM] WARNING: Unsupported arm action space type {type(arm_space)}", file=sys.stderr, flush=True)
        return None


def map_command_to_action(command: str, env_interface):
    """
    Map web control commands to robot actions.
    Returns low_level_action dict for agent 0, or None if no action.
    """
    _ensure_action_keys(env_interface)

    # Root-level Box (single vector) action space
    if ROOT_ACTION_SPACE is not None:
        if command == 'forward':
            return _make_root_box_action(forward=1.0, turn=0.0)
        if command == 'backward':
            return _make_root_box_action(forward=-1.0, turn=0.0)
        if command == 'left':
            return _make_root_box_action(forward=0.0, turn=1.0)
        if command == 'right':
            return _make_root_box_action(forward=0.0, turn=-1.0)
        if command == 'stop':
            return _make_root_box_action(forward=0.0, turn=0.0)

        print(f"[CONTROLLABLE_SIM] Command '{command}' not supported for root Box action space", flush=True)
        return None

    actions = {}

    if command == 'forward':
        base = _make_base_action(env_interface, forward=1.0, turn=0.0)
        if base:
            actions.update(base)
    elif command == 'backward':
        base = _make_base_action(env_interface, forward=-1.0, turn=0.0)
        if base:
            actions.update(base)
    elif command == 'left':
        base = _make_base_action(env_interface, forward=0.0, turn=1.0)
        if base:
            actions.update(base)
    elif command == 'right':
        base = _make_base_action(env_interface, forward=0.0, turn=-1.0)
        if base:
            actions.update(base)
    elif command == 'stop':
        base = _make_base_action(env_interface, forward=0.0, turn=0.0)
        if base:
            actions.update(base)
    elif command == 'arm_up':
        arm = _make_arm_action(env_interface, joint_index=1, amount=0.1)
        if arm:
            actions.update(arm)
    elif command == 'arm_down':
        arm = _make_arm_action(env_interface, joint_index=1, amount=-0.1)
        if arm:
            actions.update(arm)
    elif command == 'grip_open':
        arm = _make_arm_action(env_interface, joint_index=-1, amount=-1.0)
        if arm:
            actions.update(arm)
    elif command == 'grip_close':
        arm = _make_arm_action(env_interface, joint_index=-1, amount=1.0)
        if arm:
            actions.update(arm)
    else:
        print(f"[CONTROLLABLE_SIM] Unknown command received: {command}", file=sys.stderr, flush=True)
        return None

    if not actions:
        print(f"[CONTROLLABLE_SIM] No actions generated for command '{command}' (maybe action key missing?)", file=sys.stderr, flush=True)
        return None

    return {0: actions}


def _ensure_nav_skill_ready():
    """
    Reset and sanity-check the nav skill before use.
    """
    global nav_skill
    if nav_skill is None:
        return False
    try:
        nav_skill.reset([0])
        if nav_skill.prev_actions is not None:
            nav_skill.prev_actions.zero_()
        if nav_skill.not_done_masks is not None:
            nav_skill.not_done_masks.fill_(True)
        nav_skill.failed = False
        nav_skill.finished = False
        if hasattr(nav_skill, "_has_reached_goal"):
            nav_skill._has_reached_goal = torch.zeros_like(nav_skill._has_reached_goal)
        return True
    except Exception as exc:
        print(f"[NAV_SKILL] Reset failed; disabling nav-assisted movement: {exc}", file=sys.stderr, flush=True)
        nav_skill = None
        return False


def _set_nav_target(direction: str):
    """
    Configure the navigation skill to move a short distance in the desired direction.
    """
    global nav_skill
    if nav_skill is None:
        return False

    agent = nav_skill.articulated_agent
    base_T = agent.base_transformation
    base_pos = np.array(agent.base_pos)
    forward_vec = np.array(base_T.transform_vector(np.array([1.0, 0.0, 0.0])))
    right_vec = np.array(base_T.transform_vector(np.array([0.0, 0.0, -1.0])))

    forward_offset = 0.0
    lateral_offset = 0.0
    yaw_delta = 0.0

    if direction == "forward":
        forward_offset = NAV_FORWARD_DISTANCE
    elif direction == "backward":
        forward_offset = -NAV_FORWARD_DISTANCE
    elif direction == "left":
        lateral_offset = NAV_TURN_DISTANCE
        yaw_delta = np.deg2rad(NAV_TURN_ANGLE_DEG)
    elif direction == "right":
        lateral_offset = -NAV_TURN_DISTANCE
        yaw_delta = -np.deg2rad(NAV_TURN_ANGLE_DEG)
    else:
        return False

    target_pos = base_pos + forward_offset * forward_vec + lateral_offset * right_vec
    nav_skill.target_base_pos = target_pos.tolist()
    nav_skill.target_pos = nav_skill.target_base_pos

    current_rot = float(base_T.rotation().angle())
    nav_skill.target_base_rot = current_rot + yaw_delta
    nav_skill.target_is_set = True
    nav_skill.target_handle = None
    nav_skill.failed = False
    nav_skill.finished = False
    return True


def _nav_skill_step(direction: str, env_interface, observations):
    """
    Execute a short navigation sequence using the nav skill.
    Returns (handled: bool, latest observations).
    """
    if nav_skill is None:
        return False, observations

    if not _ensure_nav_skill_ready():
        return False, observations

    if not _set_nav_target(direction):
        return False, observations

    latest_obs = observations
    try:
        for step_idx in range(NAV_MAX_STEPS_PER_COMMAND):
            skill_action, _ = nav_skill.get_low_level_action(latest_obs)

            linear_idx = getattr(nav_skill, "linear_velocity_index", None)
            angular_idx = getattr(nav_skill, "angular_velocity_index", None)
            if linear_idx is None or angular_idx is None:
                break

            forward_vel = float(skill_action[linear_idx])
            turn_vel = float(skill_action[angular_idx])

            if abs(forward_vel) < 1e-4 and abs(turn_vel) < 1e-4:
                break

            base_action = _make_base_action(env_interface, forward=forward_vel, turn=turn_vel)
            if not base_action:
                break

            obs, reward, done, info = env_interface.step({0: base_action})
            latest_obs = env_interface.parse_observations(obs)
            send_current_rgb_frame(env_interface, obs_dict=obs, label=f"nav-step {direction}")

            if hasattr(nav_skill, "_has_reached_goal"):
                reached = bool(nav_skill._has_reached_goal[0].item()) if nav_skill._has_reached_goal.numel() else False
                if reached:
                    break

        nav_skill.reset([0])
        return True, latest_obs
    except Exception as exc:
        print(f"[NAV_SKILL] Error executing nav-assisted movement '{direction}': {exc}", file=sys.stderr, flush=True)
        try:
            nav_skill.reset([0])
        except Exception:
            pass
        return False, observations

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

    # Initialize navigation skill for assisted manual control
    global nav_skill
    nav_skill = None
    try:
        eval_runner = CentralizedEvaluationRunner(config.evaluation, env_interface)
        planner_agent = eval_runner.planner.agents[0]
        explore_tool = planner_agent.tools.get("Explore")
        if explore_tool and hasattr(explore_tool, "nav_skill"):
            nav_skill = explore_tool.nav_skill
            nav_skill.reset([0])
            print("[NAV_SKILL] Navigation skill ready for manual commands", file=sys.stderr, flush=True)
        else:
            print("[NAV_SKILL] Explore tool or nav skill unavailable; using raw velocity controls", file=sys.stderr, flush=True)
    except Exception as nav_exc:
        print(f"[NAV_SKILL] Failed to setup navigation skill, using raw velocity controls: {nav_exc}", file=sys.stderr, flush=True)
        nav_skill = None
    
    # Reset cached action keys (in case script is reloaded)
    global BASE_ACTION_KEY, BASE_ACTION_SUBKEY, ARM_ACTION_KEY, ARM_ACTION_SUBKEY
    BASE_ACTION_KEY = None
    BASE_ACTION_SUBKEY = None
    ARM_ACTION_KEY = None
    ARM_ACTION_SUBKEY = None
    _ensure_action_keys(env_interface)
    
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
    
    # Send initial frame immediately
    send_current_rgb_frame(env_interface, obs_dict=raw_obs, label="initial")
    
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
                if wait_count % 10 == 0:  # roughly every second
                    send_current_rgb_frame(env_interface, label="idle")
                if wait_count % 100 == 0:  # Log every 10 seconds (100 * 0.1s)
                    print(f"[CONTROLLABLE_SIM] Still waiting for commands... (waited {wait_count * 0.1:.1f}s)", file=sys.stderr, flush=True)
                time.sleep(0.1)  # 100ms polling
        
        # We have a command!
        print(f"[CONTROLLABLE_SIM] =========================================", file=sys.stderr, flush=True)
        print(f"[CONTROLLABLE_SIM] Received command: {command}", file=sys.stderr, flush=True)
        
        handled = False
        if command in {"forward", "backward", "left", "right"}:
            handled, observations = _nav_skill_step(command, env_interface, observations)
            if handled:
                step_count += 1
                print(f"[CONTROLLABLE_SIM] Executed nav-skill command: {command} (step {step_count})", file=sys.stderr, flush=True)

        if not handled:
            low_level_action = map_command_to_action(command, env_interface)
            if low_level_action:
                # Apply the action - this will step the simulation
                obs, reward, done, info = env_interface.step(low_level_action)
                observations = env_interface.parse_observations(obs)
                step_count += 1
                
                # Send updated frame directly from returned obs
                send_current_rgb_frame(env_interface, obs_dict=obs, label=f"after {command}")
                
                print(f"[CONTROLLABLE_SIM] Executed command: {command} (step {step_count})", file=sys.stderr, flush=True)
            else:
                print(f"[CONTROLLABLE_SIM] Unknown command: {command}", file=sys.stderr, flush=True)
    
    print("[CONTROLLABLE_SIM] Simulation completed", file=sys.stderr, flush=True)
    env_interface.sim.close()

if __name__ == "__main__":
    cprint("\nStarting controllable simulation", "blue")
    run_controllable_simulation()
    cprint("\nEnd of controllable simulation", "blue")

