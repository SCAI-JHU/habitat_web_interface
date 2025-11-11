#!/usr/bin/env python3
"""
Control Spot/Stretch robots directly - minimal version without world graph.
Run from project root as: python -m habitat_llm.examples.robot_control
"""

import os
import numpy as np

from habitat_llm.utils import cprint, setup_config
from habitat_llm.utils.core import get_config
from habitat_llm.agent.env import register_actions, register_sensors, EnvironmentInterface
from habitat_llm.agent.env.dataset import CollaborationDatasetV0


def control_robot(robot_name='spot', num_steps=50):
    """
    Control robot directly without perception/world graph initialization.
    
    Args:
        robot_name: 'spot' or 'stretch'
        num_steps: Number of control steps
    """
    cprint("=" * 70, "blue")
    cprint(f"ROBOT CONTROL: {robot_name.upper()}", "blue")
    cprint("=" * 70, "blue")
    
    # Get project root
    project_root = os.getcwd()
    
    # Seed
    seed = 47668090
    
    # GPU setup
    gpu_device_id = os.environ.get("HABITAT_GPU_ID", "0")
    print(f"Using GPU device ID: {gpu_device_id}")
    
    # Robot setup
    if robot_name.lower() == 'spot':
        robot_urdf_path = os.path.join(project_root, "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf")
        robot_type = "SpotRobot"
        action_name = "base_velocity_non_cylinder"
    else:
        robot_urdf_path = os.path.join(project_root, "data/robots/hab_stretch/urdf/hab_stretch.urdf")
        robot_type = "StretchRobot"
        action_name = "base_velocity"
    
    cprint(f"Robot type: {robot_type}", "green")
    cprint(f"Action: {action_name}", "green")
    
    # Config overrides
    DATASET_OVERRIDES = [
        "habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz",
        "habitat.dataset.scenes_dir=data/hssd-hab/",
        f"habitat.simulator.agents.main_agent.articulated_agent_urdf={robot_urdf_path}",
        f"habitat.simulator.agents.main_agent.articulated_agent_type={robot_type}",
        f"habitat.simulator.habitat_sim_v0.gpu_device_id={gpu_device_id}",
    ]
    
    SENSOR_OVERRIDES = [
        "habitat.simulator.agents.main_agent.sim_sensors.jaw_depth_sensor.normalize_depth=False"
    ]
    
    LLM_OVERRIDES = [
        "llm@evaluation.planner.plan_config.llm=mock",
    ]
    
    TRAJECTORY_OVERRIDES = [
        "evaluation.save_video=False",
        "evaluation.output_dir=./outputs",
        "trajectory.save=False",
        "trajectory.agent_names=[main_agent]",
    ]
    
    EPISODE_OVERRIDES = ["+episode_indices=[0]"]
    
    cprint("\nLoading config...", "yellow")
    config_base = get_config(
        "examples/single_agent_scene_mapping.yaml",
        overrides=DATASET_OVERRIDES
        + SENSOR_OVERRIDES
        + LLM_OVERRIDES
        + TRAJECTORY_OVERRIDES
        + EPISODE_OVERRIDES,
    )
    config = setup_config(config_base, seed)
    
    if config is None:
        cprint("❌ Failed to setup config", "red")
        return
    
    cprint("✅ Config loaded", "green")
    
    # Register sensors and actions
    register_sensors(config)
    register_actions(config)
    
    # Create dataset
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    if config.get("episode_indices", None) is not None:
        episode_subset = [dataset.episodes[x] for x in config.episode_indices]
        dataset = CollaborationDatasetV0(
            config=config.habitat.dataset, episodes=episode_subset
        )
    
    # Create environment WITHOUT world graph (init_wg=False to avoid snap_point error)
    cprint("Creating environment (no world graph)...", "yellow")
    env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)
    
    cprint("✅ Environment created!", "green")
    
    # Manual reset without reinitializing world graph
    env_interface.env.reset()
    
    # Get robot agent
    agent = env_interface.sim.articulated_agent
    start_pos = agent.base_pos
    
    print(f"\n📍 Starting position: [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")
    print(f"\n🔍 Action space: {env_interface.env.action_space}")
    print(f"🔍 Action space keys: {list(env_interface.env.action_space.spaces.keys()) if hasattr(env_interface.env.action_space, 'spaces') else 'N/A'}")
    cprint(f"\n🤖 Running {num_steps} control steps...\n", "blue")
    
    # Movement patterns
    movements = [
        ("Forward", [1.0, 0.0], 10),
        ("Turn Left", [0.0, 1.0], 5),
        ("Forward", [1.0, 0.0], 10),
        ("Turn Right", [0.0, -1.0], 5),
        ("Backward", [-0.5, 0.0], 5),
        ("Circle", [0.5, 0.5], 10),
        ("Stop", [0.0, 0.0], 5),
    ]
    
    step_count = 0
    for movement_name, velocity, duration in movements:
        if step_count >= num_steps:
            break
            
        cprint(f">>> {movement_name} (velocity: {velocity}) <<<", "yellow")
        
        for i in range(duration):
            if step_count >= num_steps:
                break
                
            # Create action - flat numpy array of 39 floats
            # Action space is Box(39,) where indices 7-8 are base velocity
            action_vector = np.zeros(39, dtype=np.float32)
            action_vector[7] = velocity[0]  # Linear velocity
            action_vector[8] = velocity[1]  # Angular velocity
            
            # Step environment
            try:
                obs, reward, done, info = env_interface.env.step(action_vector)
            except Exception as e:
                cprint(f"  Error during step: {e}", "red")
                break
            
            step_count += 1
            
            # Print position every 5 steps
            if step_count % 5 == 0:
                current_pos = agent.base_pos
                print(f"  Step {step_count:3d}: Pos=[{current_pos[0]:6.2f}, {current_pos[1]:6.2f}, {current_pos[2]:6.2f}]")
        
        print()
    
    # Final position
    final_pos = agent.base_pos
    distance_traveled = np.linalg.norm(final_pos - start_pos)
    
    cprint("=" * 70, "blue")
    print(f"📍 Final position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
    print(f"📏 Distance traveled: {distance_traveled:.3f} meters")
    cprint(f"✅ Completed {step_count} steps successfully!", "green")
    cprint("=" * 70, "blue")
    
    env_interface.sim.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Control robots")
    parser.add_argument('--robot', type=str, default='spot', choices=['spot', 'stretch'],
                        help='Robot to control (default: spot)')
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of steps to run (default: 50)')
    
    args = parser.parse_args()
    
    cprint("\n🤖 Starting robot control...\n", "blue")
    
    try:
        control_robot(robot_name=args.robot, num_steps=args.steps)
    except Exception as e:
        cprint(f"\n❌ Error: {e}", "red")
        import traceback
        traceback.print_exc()
    
    cprint("\n🤖 Robot control finished!\n", "blue")

