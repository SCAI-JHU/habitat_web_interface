import argparse
import os

from moviepy.editor import ImageSequenceClip


def create_video_from_frames(image_dir, output_file, fps=24):
    """
    Creates a video from a directory of sorted image frames.
    """
    print(f"Looking for image frames in: {image_dir}")

    # Get all file paths for PNG or JPG images
    image_files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith((".png", ".jpg"))
    ]

    # IMPORTANT: Sort the files alphabetically/numerically to ensure correct order
    image_files.sort()

    if not image_files:
        print("Error: No image files found in the specified directory.")
        return

    print(f"Found {len(image_files)} frames. Creating video...")

    # Create the video clip from the image sequence
    clip = ImageSequenceClip(image_files, fps=fps)

    # Write the video file to disk
    clip.write_videofile(output_file, codec="libx264")

    print(f"Successfully created video: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a video from a sequence of image frames."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="The directory containing the image frames.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="simulation_video.mp4",
        help="The path for the output video file.",
    )
    parser.add_argument(
        "--fps", type=int, default=24, help="Frames per second for the output video."
    )

    args = parser.parse_args()

    create_video_from_frames(args.image_dir, args.output_file, args.fps)
