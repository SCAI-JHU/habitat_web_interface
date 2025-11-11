#!/usr/bin/env python3
"""
Interactive keyboard control for Spot/Stretch robots.
Run from project root as: python -m habitat_llm.examples.robot_keyboard_control

Controls:
  W/S - Forward/Backward
  A/D - Turn Left/Right
  Q   - Quit
  SPACE - Stop
"""

import os
import sys
import numpy as np
import termios
import tty
import imageio
from datetime import datetime

from habitat_llm.utils import cprint, setup_config
from habitat_llm.utils.core import get_config
from habitat_llm.agent.env import register_actions, register_sensors, EnvironmentInterface
from habitat_llm.agent.env.dataset import CollaborationDatasetV0


def getch():
    """Get a single character from stdin without requiring Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def keyboard_control(robot_name='spot', save_frames=True, camera_preference='third'):
    """
    Interactive keyboard control for the robot.
    
    Args:
        robot_name: 'spot' or 'stretch'
        save_frames: Whether to save RGB frames
        camera_preference: Preferred camera ('third', 'arm', 'jaw', 'head', or 'auto')
    """
    cprint("=" * 70, "blue")
    cprint(f"INTERACTIVE KEYBOARD CONTROL: {robot_name.upper()}", "blue")
    cprint("=" * 70, "blue")
    
    # Create directory for saving frames
    if save_frames:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_dir = os.path.join("data", "robot_control_recordings", f"{robot_name}_{timestamp}", "rgb")
        os.makedirs(frame_dir, exist_ok=True)
        cprint(f"📹 Recording frames to: {frame_dir}", "green")
    
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
    else:
        robot_urdf_path = os.path.join(project_root, "data/robots/hab_stretch/urdf/hab_stretch.urdf")
        robot_type = "StretchRobot"
    
    cprint(f"Robot type: {robot_type}", "green")
    
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
    config = setup_config(config_base, seed)
    
    if config is None:
        cprint("❌ Failed to setup config", "red")
        return
    
    register_sensors(config)
    register_actions(config)
    
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    if config.get("episode_indices", None) is not None:
        episode_subset = [dataset.episodes[x] for x in config.episode_indices]
        dataset = CollaborationDatasetV0(config=config.habitat.dataset, episodes=episode_subset)
    
    env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)
    env_interface.env.reset()
    
    agent = env_interface.sim.articulated_agent
    
    cprint("✅ Environment ready!", "green")
    
    # Print controls
    print("\n" + "=" * 70)
    cprint("KEYBOARD CONTROLS:", "yellow")
    print("  W - Forward")
    print("  S - Backward")
    print("  A - Turn Left")
    print("  D - Turn Right")
    print("  SPACE - Stop")
    print("  P - Print current position")
    print("  Q - Quit")
    print("=" * 70)
    
    start_pos = agent.base_pos
    print(f"\n📍 Starting position: [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")
    
    # Get initial observation to see what cameras are available
    initial_obs = env_interface.get_observations()
    rgb_sensors = [k for k in initial_obs.keys() if 'rgb' in k.lower()]
    print(f"\n📷 Available RGB sensors: {rgb_sensors}")
    
    # Determine which camera to use based on preference
    camera_key = None
    
    if camera_preference == 'auto':
        # Auto mode: prefer third, then arm, then first available
        for key in rgb_sensors:
            if 'third_rgb' in key:
                camera_key = key
                break
        if not camera_key:
            for key in rgb_sensors:
                if 'arm_rgb' in key or 'jaw_rgb' in key:
                    camera_key = key
                    break
    else:
        # Try to match the preference
        search_terms = {
            'third': ['third_rgb'],
            'arm': ['arm_rgb'],
            'jaw': ['jaw_rgb'],
            'head': ['head', 'stereo']
        }
        
        for term in search_terms.get(camera_preference, [camera_preference]):
            for key in rgb_sensors:
                if term in key.lower():
                    camera_key = key
                    break
            if camera_key:
                break
    
    # Fallback to first available
    if camera_key is None and rgb_sensors:
        camera_key = rgb_sensors[0]
    
    print(f"📹 Using camera: {camera_key}")
    
    step_count = 0
    running = True
    current_velocity = [0.0, 0.0]
    
    cprint("\n🎮 Ready for input! Press keys to control the robot...\n", "green")
    
    try:
        while running:
            # Get keyboard input
            key = getch().lower()
            
            # Map keys to velocities
            if key == 'w':
                current_velocity = [1.0, 0.0]
                print("⬆️  Forward")
            elif key == 's':
                current_velocity = [-0.5, 0.0]
                print("⬇️  Backward")
            elif key == 'a':
                current_velocity = [0.0, 1.0]
                print("⬅️  Turn Left")
            elif key == 'd':
                current_velocity = [0.0, -1.0]
                print("➡️  Turn Right")
            elif key == ' ':
                current_velocity = [0.0, 0.0]
                print("⏸️  Stop")
            elif key == 'p':
                pos = agent.base_pos
                print(f"📍 Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                continue  # Don't step, just print
            elif key == 'q':
                print("👋 Quitting...")
                running = False
                break
            elif key == '\x03':  # Ctrl+C
                running = False
                break
            else:
                print(f"❓ Unknown key: {repr(key)}")
                continue
            
            # Create action vector
            action_vector = np.zeros(39, dtype=np.float32)
            action_vector[7] = current_velocity[0]  # Linear
            action_vector[8] = current_velocity[1]  # Angular
            
            # Execute action
            try:
                obs, reward, done, info = env_interface.env.step(action_vector)
                step_count += 1
                
                # Save frame if recording
                if save_frames and camera_key and camera_key in obs:
                    frame = obs[camera_key]
                    
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
                    
                    # Save frame
                    frame_path = os.path.join(frame_dir, f"{step_count:05d}.png")
                    imageio.imwrite(frame_path, frame)
                    
                    if step_count == 1:
                        print(f"  ✅ Saved first frame from {camera_key} to {frame_path}")
                
                # Print position every 10 steps
                if step_count % 10 == 0:
                    pos = agent.base_pos
                    print(f"  Step {step_count}: Pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
                    
            except Exception as e:
                cprint(f"❌ Error during step: {e}", "red")
                break
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    
    # Final stats
    final_pos = agent.base_pos
    distance = np.linalg.norm(final_pos - start_pos)
    
    print("\n" + "=" * 70)
    print(f"📍 Final position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
    print(f"📏 Distance traveled: {distance:.3f} meters")
    print(f"🔢 Total steps: {step_count}")
    
    if save_frames and step_count > 0:
        print(f"📹 Saved {step_count} frames to: {frame_dir}")
        print(f"\n💡 To create a video, run:")
        print(f"   ffmpeg -framerate 10 -i {frame_dir}/%05d.png -c:v libx264 -pix_fmt yuv420p {frame_dir}/../video.mp4")
    
    cprint("✅ Session complete!", "green")
    print("=" * 70)
    
    env_interface.sim.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive keyboard control for robots")
    parser.add_argument('--robot', type=str, default='spot', choices=['spot', 'stretch'],
                        help='Robot to control (default: spot)')
    parser.add_argument('--camera', type=str, default='third', 
                        choices=['third', 'arm', 'jaw', 'head', 'auto'],
                        help='Camera to use: third (high 3rd person), arm (arm camera), jaw (jaw camera), head (head stereo), auto (default: third)')
    parser.add_argument('--save-frames', action='store_true', default=True,
                        help='Save RGB frames (default: True)')
    parser.add_argument('--no-save-frames', dest='save_frames', action='store_false',
                        help='Disable frame saving')
    
    args = parser.parse_args()
    
    cprint("\n🎮 Starting interactive keyboard control...\n", "blue")
    
    try:
        keyboard_control(robot_name=args.robot, save_frames=args.save_frames, camera_preference=args.camera)
    except Exception as e:
        cprint(f"\n❌ Error: {e}", "red")
        import traceback
        traceback.print_exc()
    
    cprint("\n🎮 Keyboard control finished!\n", "blue")

