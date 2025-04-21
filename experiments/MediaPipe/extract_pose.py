import cv2
import mediapipe as mp
import json
import os
import subprocess
import shutil
import sys

def get_video_properties(video_path):
    """
    Get detailed properties of the video, including orientation.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        dict: Dictionary of video properties
    """
    properties = {}
    
    # Use OpenCV to get basic properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path} to get properties")
        return properties
    
    properties['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    properties['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    properties['fps'] = cap.get(cv2.CAP_PROP_FPS)
    properties['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Use ffprobe to get detailed metadata
    try:
        # Get full video information from ffprobe
        cmd = [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        
        print(f"Running ffprobe for detailed information: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            try:
                import json
                info = json.loads(result.stdout)
                # Extract relevant information
                if 'streams' in info and len(info['streams']) > 0:
                    video_stream = next((s for s in info['streams'] if s.get('codec_type') == 'video'), None)
                    if video_stream:
                        properties['codec'] = video_stream.get('codec_name', 'unknown')
                        properties['pix_fmt'] = video_stream.get('pix_fmt', 'unknown')
                        # Check for rotation in tags
                        if 'tags' in video_stream:
                            tags = video_stream['tags']
                            properties['tags'] = tags
                            if 'rotate' in tags:
                                properties['rotation'] = int(tags['rotate'])
                            
                        # Check for display matrix and other rotation indicators
                        if 'side_data_list' in video_stream:
                            side_data = video_stream['side_data_list']
                            for data in side_data:
                                if 'rotation' in data:
                                    properties['rotation_side_data'] = data['rotation']
            except json.JSONDecodeError:
                print(f"Error: Could not parse ffprobe output as JSON")
            except Exception as e:
                print(f"Error extracting metadata: {str(e)}")
    except Exception as e:
        print(f"Error running ffprobe: {str(e)}")
    
    # Print all the properties for debugging
    print("\nVIDEO PROPERTIES:")
    for key, value in properties.items():
        print(f"  {key}: {value}")
    print()
    
    return properties

def get_video_rotation(video_path):
    """
    Get rotation metadata from the input video file using multiple methods.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        int: Rotation angle (0, 90, 180, or 270) or 0 if not found
    """
    properties = get_video_properties(video_path)
    
    # Get rotation from properties, with fallbacks
    rotation = properties.get('rotation', None)
    if rotation is not None:
        print(f"Detected video rotation from rotate tag: {rotation} degrees")
        return rotation
    
    rotation = properties.get('rotation_side_data', None)
    if rotation is not None:
        print(f"Detected video rotation from side data: {rotation} degrees")
        return rotation
    
    # Try other methods - direct ffprobe queries
    try:
        # Check for rotate tag
        rotation_cmd = [
            "ffprobe", 
            "-v", "error", 
            "-select_streams", "v:0", 
            "-show_entries", "stream_tags=rotate", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            video_path
        ]
        
        print(f"Running explicit rotation check: {' '.join(rotation_cmd)}")
        result = subprocess.run(rotation_cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            rotation = int(result.stdout.strip())
            print(f"Detected video rotation from explicit query: {rotation} degrees")
            return rotation
        
        # Look for display matrix in side data
        matrix_cmd = [
            "ffprobe", 
            "-v", "error", 
            "-select_streams", "v:0", 
            "-show_entries", "stream_side_data_list", 
            "-of", "json", 
            video_path
        ]
        
        print(f"Running side data check: {' '.join(matrix_cmd)}")
        result = subprocess.run(matrix_cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            try:
                side_data = json.loads(result.stdout)
                if 'streams' in side_data and side_data['streams'] and 'side_data_list' in side_data['streams'][0]:
                    for data in side_data['streams'][0]['side_data_list']:
                        if 'rotation' in data:
                            rotation = data['rotation']
                            print(f"Detected rotation from side data JSON: {rotation} degrees")
                            return rotation
            except json.JSONDecodeError:
                print("Could not parse side data JSON")
            except Exception as e:
                print(f"Error checking side data: {str(e)}")
    except Exception as e:
        print(f"Error with manual rotation detection: {str(e)}")
    
    # Check dimensions as a heuristic for mobile video
    # Mobile videos are often in portrait orientation (height > width)
    if properties.get('height', 0) > properties.get('width', 0):
        print("Video has portrait orientation (height > width), might need 90° rotation")
    
    print("No rotation metadata found, assuming 0 degrees")
    return 0

def extract_pose_landmarks(video_path, save_video=True, output_video_path=None, compress=True, force_rotation=None):
    """
    Extracts 3D pose landmarks from a video file using MediaPipe.

    Args:
        video_path (str): The path to the input video file (e.g., 'serve.mov').
        save_video (bool): Whether to save the video with landmarks drawn.
        output_video_path (str): Path for the output video. If None, will use input name + "_landmarks".
        compress (bool): Whether to compress the output video using ffmpeg.
        force_rotation (int, optional): Force a specific rotation angle (0, 90, 180, 270)

    Returns:
        list: A list of dictionaries, where each dictionary contains the
              3D landmarks for a frame. Returns None if video cannot be opened.
    """
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Get the rotation of the input video
    detected_rotation = get_video_rotation(video_path)
    rotation = force_rotation if force_rotation is not None else detected_rotation
    print(f"Using rotation: {rotation} degrees (detected: {detected_rotation}, forced: {force_rotation is not None})")

    pose = mp_pose.Pose(static_image_mode=False,     # Process as video stream
                        model_complexity=2,          # Use the heavy model for better accuracy
                        enable_segmentation=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    # Get video properties for output video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Input video dimensions: {width}x{height}, FPS: {fps}")
    
    # Set up paths
    if output_video_path is None:
        # Create output video name based on input file
        filename, ext = os.path.splitext(video_path)
        output_video_path = f"{filename}_landmarks{ext}"
    
    # Define temp files for processing
    temp_video_path = output_video_path + ".temp.mp4"
    rotated_temp_path = output_video_path + ".rotated.mp4"
    
    print(f"Output path: {output_video_path}")
    print(f"Temp path: {temp_video_path}")
    
    # Check if the rotated temp path already exists and remove it
    for tmp_path in [temp_video_path, rotated_temp_path]:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                print(f"Removed existing temp file: {tmp_path}")
            except:
                print(f"Warning: Could not remove existing temp file: {tmp_path}")
    
    # Set up video writer if saving video
    video_writer = None
    if save_video:
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1'
        video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        if not video_writer.isOpened():
            print(f"Error: Could not create video writer for {temp_video_path}")
            save_video = False

    frame_pose_data = []
    frame_count = 0

    # Process the video frames
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        frame_count += 1
        if frame_count % 30 == 0:  # Print status every 30 frames
            print(f"Processing frame {frame_count}...")

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Process the image to get pose landmarks
        results = pose.process(image_rgb)

        # Store the landmark data for the current frame
        if results.pose_world_landmarks:
            frame_landmarks = []
            for i, landmark in enumerate(results.pose_world_landmarks.landmark):
                frame_landmarks.append({
                    'id': i,
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            frame_pose_data.append(frame_landmarks)
        else:
            # Append empty list if no pose is detected
            frame_pose_data.append([])

        # Draw landmarks and write to output video
        image_rgb.flags.writeable = True
        annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Write the frame to the output video if requested
        if save_video and video_writer is not None:
            video_writer.write(annotated_image)

        # Optional: Display the video with landmarks (for debugging)
        cv2.imshow('MediaPipe Pose', annotated_image)
        if cv2.waitKey(5) & 0xFF == 27:  # Press Esc to exit display
            break

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    print(f"Processed {frame_count} frames")
    
    # If we saved the video, now handle the orientation and compression
    if save_video and os.path.exists(temp_video_path):
        # Try multiple approaches to fix orientation
        success = False
        
        # First, try using metadata
        print("\nTrying orientation correction with metadata...")
        if correct_video_orientation_with_metadata(temp_video_path, output_video_path, rotation):
            success = True
        
        # If that fails, try using physical rotation
        if not success:
            print("\nTrying physical rotation of video frames...")
            if correct_video_orientation_with_transform(temp_video_path, rotated_temp_path, rotation):
                # If successful, move the rotated temp to the output path
                try:
                    shutil.move(rotated_temp_path, output_video_path)
                    success = True
                    print(f"Moved transformed video to {output_video_path}")
                except Exception as e:
                    print(f"Error moving transformed video: {str(e)}")
        
        # If both fail, copy the original temp file as fallback
        if not success:
            print("\nFalling back to original orientation...")
            try:
                shutil.copy2(temp_video_path, output_video_path)
                print(f"Copied unmodified video to {output_video_path}")
            except Exception as e:
                print(f"Error copying fallback video: {str(e)}")
        
        # Remove temporary files
        for tmp_path in [temp_video_path, rotated_temp_path]:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                    print(f"Removed temp file: {tmp_path}")
                except:
                    print(f"Warning: Could not remove temp file: {tmp_path}")
        
        print(f"\nVideo with landmarks saved to {output_video_path}")
        
        # Verify the orientation of the final video
        print("\nChecking orientation of output video:")
        output_rotation = get_video_rotation(output_video_path)
        print(f"Output video rotation: {output_rotation} degrees")
        
        # Compress the output video with ffmpeg if requested
        if compress and os.path.exists(output_video_path):
            compress_video(video_path, output_video_path)
    
    return frame_pose_data

def correct_video_orientation_with_metadata(input_path, output_path, rotation):
    """
    Use ffmpeg to copy the video with the correct orientation metadata.
    
    Args:
        input_path (str): Path to the input video file
        output_path (str): Path for the output video file
        rotation (int): Rotation angle in degrees (0, 90, 180, or 270)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create the ffmpeg command to copy the video with rotation metadata
        if rotation == 0:
            # Simple copy without re-encoding if no rotation is needed
            cmd = [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-c", "copy",
                output_path
            ]
        else:
            # Add rotation metadata - try both methods
            cmd = [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-c:v", "libx264",
                "-metadata:s:v:0", f"rotate={rotation}",
                "-tag:v", "avc1", # Ensure correct tagging
                "-preset", "medium",
                output_path
            ]
        
        print(f"Running ffmpeg to correct orientation with metadata: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            print(f"Error correcting orientation with metadata: {result.stderr}")
            return False
        
        # Check if output file exists and has correct properties
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print("Orientation correction with metadata appears successful")
            return True
        else:
            print("Orientation correction failed - output file missing or empty")
            return False
    except Exception as e:
        print(f"Exception in orientation correction with metadata: {str(e)}")
        return False

def correct_video_orientation_with_transform(input_path, output_path, rotation):
    """
    Use ffmpeg to physically rotate the video frames.
    
    Args:
        input_path (str): Path to the input video file
        output_path (str): Path for the output video file
        rotation (int): Rotation angle in degrees (0, 90, 180, or 270)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Skip if no rotation needed
        if rotation == 0:
            print("No physical rotation needed, using copy")
            shutil.copy2(input_path, output_path)
            return True
        
        # Map rotation angle to ffmpeg transpose parameter
        transpose_param = None
        if rotation == 90:
            transpose_param = "1"  # 90 degrees clockwise
        elif rotation == 180:
            transpose_param = "2,2"  # Apply twice for 180 degrees
        elif rotation == 270:
            transpose_param = "2"  # 90 degrees counterclockwise
        
        if transpose_param is None:
            print(f"Unsupported rotation angle: {rotation}")
            return False
        
        # Build the ffmpeg command for physical rotation
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-vf", f"transpose={transpose_param}",
            "-c:v", "libx264",
            "-preset", "medium",
            output_path
        ]
        
        print(f"Running ffmpeg for physical rotation: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            print(f"Error with physical rotation: {result.stderr}")
            return False
        
        # Check if output file exists and has correct properties
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print("Physical rotation successful")
            return True
        else:
            print("Physical rotation failed - output file missing or empty")
            return False
    except Exception as e:
        print(f"Exception in physical rotation: {str(e)}")
        return False

def compress_video(original_path, landmarks_path):
    """
    Compress the landmarks video to a similar size as the original video using ffmpeg.
    
    Args:
        original_path (str): Path to the original video
        landmarks_path (str): Path to the landmarks video to compress
    """
    try:
        # Check if ffmpeg is available
        ffmpeg_check = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=False)
        if ffmpeg_check.returncode != 0:
            print("ffmpeg check failed:")
            print(ffmpeg_check.stdout)
            print(ffmpeg_check.stderr)
            print("ffmpeg not found or not working properly. Skipping compression.")
            return
        else:
            print(f"ffmpeg found: {ffmpeg_check.stdout.splitlines()[0] if ffmpeg_check.stdout else 'Unknown version'}")
    except Exception as e:
        print(f"Error checking ffmpeg: {str(e)}")
        print("ffmpeg not found. Skipping compression.")
        return
    
    # Get original file size in bytes
    original_size = os.path.getsize(original_path)
    landmarks_size = os.path.getsize(landmarks_path)
    
    print(f"Original video size: {original_size / (1024*1024):.2f} MB")
    print(f"Landmarks video size before compression: {landmarks_size / (1024*1024):.2f} MB")
    
    # Create a temporary file for the compressed output
    output_dir = os.path.dirname(landmarks_path) or "."
    temp_output = os.path.join(output_dir, "temp_compressed.mp4")
    print(f"Temp output path: {temp_output}")
    
    # Get video duration using ffprobe
    duration_cmd = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        landmarks_path
    ]
    
    print(f"Running duration command: {' '.join(duration_cmd)}")
    try:
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, check=False)
        if duration_result.returncode != 0:
            print(f"Error getting duration: {duration_result.stderr}")
            duration = 10.0  # Assume 10 seconds if we can't determine
            print(f"Using default duration: {duration} seconds")
        else:
            duration = float(duration_result.stdout.strip())
            print(f"Video duration: {duration} seconds")
    except Exception as e:
        print(f"Exception getting duration: {str(e)}")
        duration = 10.0  # Assume 10 seconds if we can't determine
        print(f"Using default duration: {duration} seconds")
    
    # Calculate target bitrate (match original file size to bitrate)
    # Formula: bitrate = filesize / duration
    target_bitrate = int((original_size * 8) / duration)  # Convert bytes to bits
    print(f"Target bitrate: {target_bitrate} bits/s ({target_bitrate/1000:.2f} kbps)")
    
    # Try simpler one-pass encoding first instead of two-pass
    compress_cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-i", landmarks_path,
        "-c:v", "libx264",
        "-b:v", f"{target_bitrate}",
        "-preset", "medium",  # Balance between compression speed and ratio
        temp_output
    ]
    
    print(f"Running ffmpeg compress command: {' '.join(compress_cmd)}")
    try:
        # Don't capture output so we can see it in real-time
        compress_result = subprocess.run(compress_cmd, capture_output=True, text=True, check=False)
        if compress_result.returncode != 0:
            print(f"ffmpeg compression failed with error code {compress_result.returncode}")
            print(f"Error: {compress_result.stderr}")
            print("Trying an alternative approach...")
            
            # Try a simpler command with constant rate factor (CRF) instead of bitrate
            alt_compress_cmd = [
                "ffmpeg",
                "-y",
                "-i", landmarks_path,
                "-c:v", "libx264",
                "-crf", "23",  # Medium quality (lower = better quality, higher = smaller file)
                "-preset", "medium",
                temp_output
            ]
            
            print(f"Running alternative ffmpeg command: {' '.join(alt_compress_cmd)}")
            alt_result = subprocess.run(alt_compress_cmd, capture_output=False, check=False)
            
            if alt_result.returncode != 0:
                print("Alternative compression also failed. Check ffmpeg installation.")
                return
        else:
            print("Compression completed successfully")
    except Exception as e:
        print(f"Exception during compression: {str(e)}")
        print("Compression failed. Using original landmarks video.")
        return
    
    # Check if the output file exists and has a reasonable size
    if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
        print(f"Temp file exists with size: {os.path.getsize(temp_output) / (1024*1024):.2f} MB")
        try:
            # Make backup of original file first
            backup_path = landmarks_path + ".backup"
            shutil.copy2(landmarks_path, backup_path)
            print(f"Created backup at {backup_path}")
            
            # Replace the original landmarks file with the compressed one
            shutil.move(temp_output, landmarks_path)
            compressed_size = os.path.getsize(landmarks_path)
            print(f"Landmarks video size after compression: {compressed_size / (1024*1024):.2f} MB")
            print(f"Compression ratio: {landmarks_size / compressed_size:.2f}x")
        except Exception as e:
            print(f"Error replacing file: {str(e)}")
            print("Compression completed but could not replace original file.")
    else:
        if os.path.exists(temp_output):
            print(f"Temp file exists but has zero size: {os.path.getsize(temp_output)} bytes")
        else:
            print(f"Temp file does not exist at {temp_output}")
        print("Compression failed. Using original landmarks video.")

if __name__ == "__main__":
    video_file = 'serve.mov' # Default input file
    
    # Check for command line arguments
    force_rotation = None
    if len(sys.argv) > 1:
        # First argument can be the video file
        video_file = sys.argv[1]
        
        # Check for rotation flag
        if len(sys.argv) > 2:
            try:
                force_rotation = int(sys.argv[2])
                print(f"Forcing rotation to {force_rotation} degrees")
            except ValueError:
                print(f"Invalid rotation value: {sys.argv[2]}")
    
    # Run the extraction with possible forced rotation
    pose_data = extract_pose_landmarks(
        video_file, 
        save_video=True, 
        compress=True,
        force_rotation=force_rotation
    )

    if pose_data:
        output_json_file = 'serve_pose_data.json'
        with open(output_json_file, 'w') as f:
            json.dump(pose_data, f, indent=4)
        print(f"Successfully extracted pose data and saved to {output_json_file}")
    else:
        print("Pose extraction failed.")