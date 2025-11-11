#!/usr/bin/env python3
"""
Refactored robot control logic into a controllable class for a web server.
This file is based on robot_keyboard_control.py.

Original controls:
  W/S - Forward/Backward
  A/D - Turn Left/Right
  SPACE - Stop
"""

import os
import sys
import numpy as np
import imageio
from datetime import datetime

# Assuming this script is in habitat_llm/examples, we need project root
# os.path.abspath(__file__) -> /.../habitat_llm/examples/robot_simulator.py
# os.path.dirname(...) -> /.../habitat_llm/examples
# os.path.dirname(...) -> /.../habitat_llm
# os.path.dirname(...) -> /.../ (project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add project root to path to allow imports
sys.path.insert(0, PROJECT_ROOT)

from habitat_llm.utils import cprint, setup_config
from habitat_llm.utils.core import get_config
from habitat_llm.agent.env import register_actions, register_sensors, EnvironmentInterface
from habitat_llm.agent.env.dataset import CollaborationDatasetV0

class RobotSimulator:
    """
    A class to wrap the Habitat simulation for external control.
    """
    def __init__(self, robot_name='stretch', camera_preference='arm'):
        cprint("=" * 70, "blue")
        cprint(f"INITIALIZING SIMULATOR: {robot_name.upper()}", "blue")
        cprint("=" * 70, "blue")

        self.project_root = PROJECT_ROOT
        self.seed = 47668090
        self.gpu_device_id = os.environ.get("HABITAT_GPU_ID", "0")
        
        # Robot setup
        if robot_name.lower() == 'spot':
            robot_urdf_path = os.path.join(self.project_root, "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf")
            robot_type = "SpotRobot"
        else:
            robot_urdf_path = os.path.join(self.project_root, "data/robots/hab_stretch/urdf/hab_stretch.urdf")
            robot_type = "StretchRobot"
        
        cprint(f"Robot type: {robot_type}", "green")
        
        # Config overrides
        DATASET_OVERRIDES = [
            "habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz",
            "habitat.dataset.scenes_dir=data/hssd-hab/",
            f"habitat.simulator.agents.main_agent.articulated_agent_urdf={robot_urdf_path}",
            f"habitat.simulator.agents.main_agent.articulated_agent_type={robot_type}",
            f"habitat.simulator.habitat_sim_v0.gpu_device_id={self.gpu_device_id}",
        ]
        SENSOR_OVERRIDES = [
            "habitat.simulator.agents.main_agent.sim_sensors.jaw_depth_sensor.normalize_depth=False"
        ]
        LLM_OVERRIDES = ["llm@evaluation.planner.plan_config.llm=mock"]
        TRAJECTORY_OVERRIDES = [
            "evaluation.save_video=False",
            "evaluation.output_dir=./outputs",
            "trajectory.save=False",
            "trajectory.agent_names=[main_agent]",
        ]
        EPISODE_OVERRIDES = ["+episode_indices=[0]"]
        
        cprint("\nLoading environment...", "yellow")
        config_base = get_config(
            "examples/single_agent_scene_mapping.yaml",
            overrides=DATASET_OVERRIDES + SENSOR_OVERRIDES + LLM_OVERRIDES + TRAJECTORY_OVERRIDES + EPISODE_OVERRIDES,
        )
        config = setup_config(config_base, self.seed)
        
        if config is None:
            cprint("❌ Failed to setup config", "red")
            raise RuntimeError("Failed to setup Habitat config")
        
        register_sensors(config)
        register_actions(config)
        
        dataset = CollaborationDatasetV0(config.habitat.dataset)
        if config.get("episode_indices", None) is not None:
            episode_subset = [dataset.episodes[x] for x in config.episode_indices]
            dataset = CollaborationDatasetV0(config=config.habitat.dataset, episodes=episode_subset)
        
        self.env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)
        self.env_interface.env.reset()
        
        self.agent = self.env_interface.sim.articulated_agent
        self.camera_key = None
        self.step_count = 0
        
        cprint("✅ Environment ready!", "green")
        start_pos = self.agent.base_pos
        print(f"\n📍 Starting position: [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")
        
        # Get initial observation to determine camera
        initial_obs = self.env_interface.get_observations()
        self._find_camera(initial_obs, camera_preference)
        
        if self.camera_key:
            print(f"📹 Using camera: {self.camera_key}")
        else:
            print(f"⚠️ No suitable camera found matching '{camera_preference}'")
            
    def _find_camera(self, obs, camera_preference):
        """Helper to find the best camera key."""
        rgb_sensors = [k for k in obs.keys() if 'rgb' in k.lower()]
        print(f"📷 Available RGB sensors: {rgb_sensors}")
        
        if camera_preference == 'auto':
            for key in rgb_sensors:
                if 'third_rgb' in key: self.camera_key = key; break
            if not self.camera_key:
                for key in rgb_sensors:
                    if 'arm_rgb' in key or 'jaw_rgb' in key: self.camera_key = key; break
        else:
            search_terms = {
                'third': ['third_rgb'],
                'arm': ['arm_rgb'],
                'jaw': ['jaw_rgb'],
                'head': ['head', 'stereo']
            }
            for term in search_terms.get(camera_preference, [camera_preference]):
                for key in rgb_sensors:
                    if term in key.lower(): self.camera_key = key; break
                if self.camera_key: break
        
        if self.camera_key is None and rgb_sensors:
            self.camera_key = rgb_sensors[0]

    def _process_frame(self, obs):
        """Extracts, processes, and returns the camera frame from an observation."""
        if not self.camera_key or self.camera_key not in obs:
            return None # Return nothing if no camera
            
        frame = obs[self.camera_key]
        
        # Handle torch tensors
        if not isinstance(frame, np.ndarray):
            frame = frame.cpu().numpy()
        
        # Convert to uint8 if needed
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # Handle RGBA
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
            
        return frame

    def get_initial_frame(self):
        """Get the very first frame."""
        obs = self.env_interface.get_observations()
        return self._process_frame(obs)

    def step(self, key):
        """
        Takes a keyboard command and steps the simulation.
        Returns a new frame.
        """
        current_velocity = [0.0, 0.0]
        
        if key == 'w':
            current_velocity = [3.0, 0.0]
        elif key == 's':
            current_velocity = [-2.0, 0.0]
        elif key == 'a':
            current_velocity = [0.0, 2.0]
        elif key == 'd':
            current_velocity = [0.0, -2.0]
        elif key == ' ':
            current_velocity = [0.0, 0.0]
        elif key == 'p':
            pos = self.agent.base_pos
            print(f"📍 Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            return self.get_initial_frame() # Don't step, just return last frame
        else:
            # Unknown key, do nothing
            return self.get_initial_frame()
            
        # Create action vector
        action_vector = np.zeros(39, dtype=np.float32)
        action_vector[7] = current_velocity[0]  # Linear
        action_vector[8] = current_velocity[1]  # Angular
        
        # Execute action
        try:
            obs, reward, done, info = self.env_interface.env.step(action_vector)
            self.step_count += 1
            
            if self.step_count % 20 == 0:
                pos = self.agent.base_pos
                print(f"  Step {self.step_count}: Pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
                
            return self._process_frame(obs)
            
        except Exception as e:
            cprint(f"❌ Error during step: {e}", "red")
            return None

    def close(self):
        """Cleans up the simulation."""
        cprint("Simulation shutting down...", "yellow")
        self.env_interface.sim.close()
        cprint("✅ Session complete!", "green")

if __name__ == "__main__":
    # You can run this file directly to test the refactored class
    cprint("Testing RobotSimulator class...", "magenta")
    
    sim = None
    try:
        sim = RobotSimulator(robot_name='stretch', camera_preference='arm')
        
        frame = sim.get_initial_frame()
        print(f"Got initial frame with shape: {frame.shape}")
        
        # Test a few steps
        print("Stepping forward...")
        frame = sim.step('w')
        print("Stepping left...")
        frame = sim.step('a')
        print("Stopping...")
        frame = sim.step(' ')
        print(f"Got final frame with shape: {frame.shape}")
        
    except Exception as e:
        cprint(f"❌ Test failed: {e}", "red")
    finally:
        if sim:
            sim.close()