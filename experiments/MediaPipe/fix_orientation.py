import os
import subprocess
import sys
import shutil

def fix_video_orientation(input_path, output_path, rotation):
    """
    Directly fixes the orientation of the video by physically rotating it
    using ffmpeg's transpose filter.
    
    Args:
        input_path: Path to the input video file
        output_path: Path for the output video file
        rotation: Rotation angle in degrees (90, 180, 270)
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Map rotation angle to ffmpeg transpose parameter
    transpose_map = {
        90: "transpose=1",                 # 90 degrees clockwise
        180: "transpose=2,transpose=2",    # 180 degrees (two 90-degree rotations)
        270: "transpose=2",                # 90 degrees counterclockwise
        "-90": "transpose=2",              # Alternative way to specify counterclockwise
    }
    
    # Get transpose parameter based on rotation
    transpose_value = transpose_map.get(str(rotation) if isinstance(rotation, str) else rotation)
    
    if not transpose_value:
        print(f"Error: Unsupported rotation angle: {rotation}")
        print(f"Supported angles: {list(transpose_map.keys())}")
        return False
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Command to perform the rotation
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-i", input_path,
        "-vf", transpose_value,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",  # Balance quality and file size
        output_path
    ]
    
    print(f"Executing command:\n{' '.join(cmd)}")
    
    try:
        # Run ffmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check if command was successful
        if result.returncode != 0:
            print("Error executing ffmpeg command:")
            print(result.stderr)
            return False
        
        print(f"Successfully created rotated video at {output_path}")
        return True
    except Exception as e:
        print(f"Error during video rotation: {str(e)}")
        return False

def main():
    # Default values
    input_video = "serve_landmarks.mov"
    output_video = "serve_landmarks_fixed.mov"
    rotation = 270  # Default to 270 degrees (counter-clockwise rotation)
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        input_video = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            rotation = int(sys.argv[2])
        except ValueError:
            print(f"Invalid rotation value: {sys.argv[2]}")
            print("Using default rotation: 270 degrees")
    if len(sys.argv) > 3:
        output_video = sys.argv[3]
    
    # Validate input file exists
    if not os.path.exists(input_video):
        print(f"Error: Input file '{input_video}' does not exist")
        return
    
    print(f"Input video: {input_video}")
    print(f"Output video: {output_video}")
    print(f"Rotation: {rotation} degrees")
    
    # Fix the orientation
    if fix_video_orientation(input_video, output_video, rotation):
        print("\nRotation successful!")
        print(f"New video saved as: {output_video}")
        
        # Ask if user wants to replace the original file
        replace = input("\nReplace original file with the fixed version? (y/n): ").lower().strip()
        if replace == 'y':
            try:
                # Backup the original file
                backup_path = input_video + ".backup"
                shutil.copy2(input_video, backup_path)
                print(f"Original file backed up to: {backup_path}")
                
                # Replace the original with the fixed version
                shutil.copy2(output_video, input_video)
                print(f"Original file replaced with the fixed version")
                
                # Option to remove the intermediate file
                remove = input("Remove the intermediate fixed file? (y/n): ").lower().strip()
                if remove == 'y':
                    os.remove(output_video)
                    print(f"Removed intermediate file: {output_video}")
            except Exception as e:
                print(f"Error replacing file: {str(e)}")
    else:
        print("Rotation failed")

if __name__ == "__main__":
    main() 