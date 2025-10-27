#!/usr/bin/env python3
# isort: skip_file

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script implements structured episodes over a collection of scenes, which
ask the agent to go to each furniture within the scene and save a RGBD+pose trajectory.
This trajectory is then used to create a map of the scenes through Concept-Graphs.
"""
import os
import sys

# for photo transmission
import base64
import io

# Create a unique episode name
import datetime

# Force unbuffered output for better debugging
import functools
print = functools.partial(print, flush=True)

# For image saving
import numpy as np
import imageio

# append the path of the
# parent directory
sys.path.append("..")

# Force reload of modules to pick up code changes - DISABLED for now
# Reloading doesn't work well with Hydra configs
# if 'habitat_llm.agent.agent' in sys.modules:
#     import importlib
#     importlib.reload(sys.modules['habitat_llm.agent.agent'])

from habitat_llm.utils import cprint, setup_config
from habitat_llm.agent.env import EnvironmentInterface, register_actions, register_sensors
from habitat_llm.evaluation import CentralizedEvaluationRunner
from habitat_llm.world_model import Room
from habitat_llm.utils.core import get_config
from habitat_llm.agent.env.dataset import CollaborationDatasetV0

def send_frame_to_stdout(frame_array):
    """
    Hello
    Encodes a frame array (NumPy) into a Base64 string and prints it.
    """
    try:
        # Ensure array is uint8
        if frame_array.dtype != np.uint8:
            if frame_array.max() <= 1.0:
                frame_array = (frame_array * 255).astype(np.uint8)
            else:
                frame_array = frame_array.astype(np.uint8)
        
        # Handle RGBA if necessary
        if frame_array.shape[2] == 4:
            frame_array = frame_array[:, :, :3]

        buffer = io.BytesIO()
        imageio.imwrite(buffer, frame_array, format='PNG') # Save image to a memory buffer
        b64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # Print the data URI format, which the browser can display directly
        # The 'flush=True' is crucial for the server to read it immediately
        print(f"data:image/png;base64,{b64_string}", flush=True) 
        
    except Exception as e:
        print(f"[ERROR_STREAM] Failed to encode/send frame: {e}", flush=True)


# --- REVISED HELPER FUNCTION TO SAVE FRAMES (Now calls send_frame_to_stdout) ---
def save_rgb_frame(frame_array, episode_dir, step_idx):
    """
    Save a single RGB frame to disk AND send it to stdout for streaming.
    
    frame_array: torch.Tensor or np.ndarray HxWx3
    episode_dir: str, folder path for this episode
    step_idx: int, current frame number
    """
    try:
        # Convert from PyTorch Tensor (on GPU) to NumPy array (on CPU)
        numpy_frame = frame_array # Default if already numpy
        if not isinstance(frame_array, np.ndarray):
            numpy_frame = frame_array.cpu().numpy()
            
        # --- Send frame for live streaming ---
        send_frame_to_stdout(numpy_frame) 
        # ---
            
        # --- Continue saving frame to disk (optional, keep for recording) ---
        rgb_dir = os.path.join(episode_dir, "rgb")
        os.makedirs(rgb_dir, exist_ok=True)
        filename = os.path.join(rgb_dir, f"{step_idx:05d}.png")
        
        if step_idx < 3:
            print(f"[TRAJ] Saving frame {step_idx} to: {filename}")
        
        # Ensure array is in the correct format (uint8)
        save_frame = numpy_frame # Use the converted numpy frame
        if save_frame.dtype != np.uint8:
            if save_frame.max() <= 1.0:
                save_frame = (save_frame * 255).astype(np.uint8)
            else:
                save_frame = save_frame.astype(np.uint8)
                
        # Handle 4-channel RGBA images if they appear
        if save_frame.shape[2] == 4:
            save_frame = save_frame[:, :, :3] # Drop the alpha channel
            
        imageio.imwrite(filename, save_frame)
        
        # Optional: Log a success message for the first few frames
        if step_idx < 5:
            # We print to stderr now to avoid interfering with stdout stream
            print(f"[DEBUG_SAVE] Successfully saved frame {step_idx} to {filename}", file=sys.stderr, flush=True) 
            
    except Exception as e:
        # Print errors to stderr
        print(f"[ERROR_SAVE] Failed to save/send frame {step_idx} to {episode_dir}: {e}", file=sys.stderr, flush=True)
# --- END HELPER FUNCTION ---


# Method to load agent planner from the config
def run_planner():
    # Get the project root directory dynamically
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    
    # NOTE: The old 'traj_dir' logic was removed.
    # A new 'traj_dir' will be created dynamically for EACH episode inside the loop.

    # Setup a seed
    seed = 47668090

    # setup required overrides
    # Get GPU device ID from environment variable or use default
    # When CUDA_VISIBLE_DEVICES is set by SLURM, device 0 refers to the first visible device
    gpu_device_id = os.environ.get("HABITAT_GPU_ID", "0")
    print(f"Using GPU device ID: {gpu_device_id}")
    
    # Construct robot URDF path dynamically
    robot_urdf_path = os.path.join(project_root, "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf")
    
    DATASET_OVERRIDES = [
        # "habitat.dataset.data_path=data/datasets/path/to/val/scenes",
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
        "trajectory.save=True",  # Enable trajectory image saving
        "trajectory.agent_names=[main_agent]",
    ]

    EPISODE_OVERRIDES = ["+episode_indices=[2]"]  # USE FOR VAL SCENES

    # Setup config
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

    # We register the dynamic habitat sensors
    register_sensors(config)

    # We register custom actions
    register_actions(config)

    # Initialize the environment interface for the agent
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    print("xytest", config.episode_indices)
    print("xytest", len(dataset.episodes))
    if config.get("episode_indices", None) is not None:
        episode_subset = [dataset.episodes[x] for x in config.episode_indices]
        dataset = CollaborationDatasetV0(
            config=config.habitat.dataset, episodes=episode_subset
        )
    env_interface = EnvironmentInterface(config, dataset=dataset)

    # Instantiate the agent planner
    # NOTE: we don't strictly need this but it has good helper functions to make
    # scripted execution easy
    eval_runner = CentralizedEvaluationRunner(config.evaluation, env_interface)

    # book-keeping and verbosity
    # Highlight the mode of operation
    cprint("\n---------------------------------------", "blue")
    cprint(f"Planner Mode: {config.evaluation.type.capitalize()}", "blue")
    # cprint(f"LLM model: {config.planner.llm.llm._target_}", "blue")
    cprint(f"Partial Observability: {config.world_model.partial_obs}", "blue")
    # Print the agent list
    cprint(f"Agent List: {eval_runner.agent_list}", "blue")
    if env_interface._single_agent_mode:
        cprint("Single agent mode", "green")
    cprint("---------------------------------------\n", "blue")
    num_episodes = len(env_interface.env.episodes)
    processed_scenes = set()
    robot_agent_uid = config.robot_agent_uid

    # initial reset to load first episode
    for idx in range(num_episodes):
        env_interface.reset_environment()
        eval_runner.reset()
        cur_episode = env_interface.env.env.env._env.current_episode
        cur_episode.episode_id = idx
        scene_id = cur_episode.scene_id
        
        # Manual trajectory saving disabled - using Habitat's built-in system only
        episode_step_count = 0

        if str(scene_id) in processed_scenes:
            print(f"Skipping scene {scene_id}. Already mapped.")
            continue
        print(
            f"Processing scene: {scene_id}, episode: {idx+1}/{num_episodes}, processed scenes: {len(processed_scenes)}/10"
        )
        if len(processed_scenes) == 10:
            break
        
        # --- START OF FIX ---
        # Get the RAW observation first
        raw_obs = env_interface.get_observations()

        # Manual frame saving disabled - Habitat handles this
        
        # NOW, parse the observation to feed to the agent
        observations = env_interface.parse_observations(raw_obs)
        # --- END OF FIX ---

        # get the list of all rooms in this house
        rooms = env_interface.world_graph[robot_agent_uid].get_all_nodes_of_type(Room)

        print(f"---Total number of rooms in this house: {len(rooms)}---\n\n")
        
        # Limit to first 2 rooms for faster testing
        rooms = rooms[:2]
        print(f"Limiting to {len(rooms)} rooms for testing\n")
        
        # For even faster testing, you can limit furniture per room
        MAX_FURNITURE_PER_ROOM = 5  # Set to None to explore all furniture
        if MAX_FURNITURE_PER_ROOM:
            print(f"[TEST MODE] Limiting to {MAX_FURNITURE_PER_ROOM} furniture items per room\n")
        
        while rooms:
            print(f"{len(rooms)} more room to go...")
            current_room = rooms.pop()
            
            # Optionally limit furniture for faster testing
            if MAX_FURNITURE_PER_ROOM:
                # Temporarily modify the world graph to limit furniture
                original_get_furniture = env_interface.world_graph[robot_agent_uid].get_furniture_in_room
                def limited_get_furniture(room_name):
                    furniture = original_get_furniture(room_name)
                    if len(furniture) > MAX_FURNITURE_PER_ROOM:
                        print(f"[TEST MODE] Room {room_name} has {len(furniture)} furniture, limiting to {MAX_FURNITURE_PER_ROOM}")
                        furniture = furniture[:MAX_FURNITURE_PER_ROOM]
                    return furniture
                env_interface.world_graph[robot_agent_uid].get_furniture_in_room = limited_get_furniture
            
            hl_action_name = "Explore"
            hl_action_input = current_room.name
            hl_action_done = False
            print(f"Executing high-level action: {hl_action_name} on {hl_action_input}")
            
            try:
                step_count = 0
                max_steps_per_room = 3000  # Safety timeout
                print(f"[SCENE_MAPPING] Starting room exploration loop for {hl_action_input}")
                while not hl_action_done:
                    step_count += 1
                    
                    # Safety timeout check
                    if step_count >= max_steps_per_room:
                        print(f"\t[WARNING] Exceeded max steps ({max_steps_per_room}) for room {hl_action_input}")
                        print(f"\t[WARNING] Forcing completion and moving to next room")
                        break
                    
                    # Get response and/or low level actions
                    print(f"[SCENE_MAPPING] Step {step_count}: About to call process_high_level_action", flush=True)
                    
                    # Direct check - what is the agent?
                    agent = eval_runner.planner.agents[0]
                    if step_count == 1:
                        print(f"[SCENE_MAPPING] Agent type: {type(agent).__name__}", flush=True)
                        print(f"[SCENE_MAPPING] Agent has process_high_level_action: {hasattr(agent, 'process_high_level_action')}", flush=True)
                    
                    try:
                        # Introspect the agent's tools
                        if step_count == 1:
                            print(f"[SCENE_MAPPING] Agent tools: {list(agent.tools.keys())}")
                            explore_tool = agent.tools.get('Explore')
                            if explore_tool:
                                print(f"[SCENE_MAPPING] Explore tool type: {type(explore_tool).__name__}")
                                print(f"[SCENE_MAPPING] Explore tool has skill: {hasattr(explore_tool, 'skill')}")
                                if hasattr(explore_tool, 'skill'):
                                    skill = explore_tool.skill
                                    print(f"[SCENE_MAPPING] Skill type: {type(skill).__name__}")
                                    print(f"[SCENE_MAPPING] Skill _cur_skill_step: {skill._cur_skill_step}")
                                    print(f"[SCENE_MAPPING] Skill target_is_set: {skill.target_is_set}")
                        
                        # Agent is fed the PARSED observations
                        low_level_action, response = agent.process_high_level_action(
                            hl_action_name, hl_action_input, observations
                        )
                        
                        # Check skill state after call
                        if step_count <= 3 or step_count % 50 == 0:
                            explore_tool = agent.tools.get('Explore')
                            if explore_tool and hasattr(explore_tool, 'skill'):
                                skill = explore_tool.skill
                                print(f"[SCENE_MAPPING] Step {step_count}: Skill step: {skill._cur_skill_step[0].item()}, finished: {skill.finished}, failed: {skill.failed}")
                                if hasattr(skill, 'target_room_name'):
                                    print(f"[SCENE_MAPPING] Step {step_count}: Room: {skill.target_room_name}, fur_queue: {len(skill.fur_queue)}, target_fur: {skill.target_fur_name}")
                        
                        print(f"[SCENE_MAPPING] Step {step_count}: Returned, response='{response}', action is None: {low_level_action is None}", flush=True)
                    except Exception as e:
                        print(f"[SCENE_MAPPING] ERROR in process_high_level_action: {type(e).__name__}: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                        raise
                    
                    # Debug output every 100 steps
                    if step_count % 100 == 0:
                        print(f"\t[DEBUG] Step {step_count}: Room={hl_action_input}, Response={response}")
                    
                    low_level_action = {0: low_level_action}
                    # 'obs' is the NEW RAW observation from the simulator
                    obs, reward, done, info = env_interface.step(low_level_action)
                    # 'observations' is the NEW PARSED observation for the *next* loop
                    observations = env_interface.parse_observations(obs)
                    
                    # Manual frame saving disabled - Habitat handles this
                    episode_step_count += 1

                    # figure out how to get completion signal
                    if response:
                        print(f"\tResponse: {response}")
                        hl_action_done = True
                print(
                    f"\tCompleted high-level action: {hl_action_name} on {hl_action_input}"
                )
            except Exception as e:
                print(f"\tError exploring {hl_action_input}: {e}")
                print(f"\tSkipping to next room...")
                continue

        # if eval_runner.frames:
        #     eval_runner._make_video(scene_id)
        processed_scenes.add(str(scene_id))

        # NOTE: Removed the old file counting logic as it's no longer relevant.
        print(f"Finished processing episode for scene {scene_id}.")

    env_interface.sim.close()


if __name__ == "__main__":
    cprint(
        "\nStart of the scene mapping routine",
        "blue",
    )

    # Run planner
    run_planner()

    cprint(
        "\nEnd of the single-agent, scene-mapping routine",
        "blue",
    )