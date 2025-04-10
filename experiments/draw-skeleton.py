import logging
import cv2 # OpenCV for video processing
from ultralytics import YOLO # YOLOv8
import numpy as np
import os
import argparse
import json
from typing import List, Dict, Any

# --- Configuration ---
POSE_MODEL_NAME = 'yolov8s-pose.pt'
OBJECT_MODEL_NAME = 'yolov8s.pt'  # Using the standard YOLO model for object detection
try:
    # Load models globally
    pose_model = YOLO(POSE_MODEL_NAME)
    object_model = YOLO(OBJECT_MODEL_NAME)
    logging.info(f"Successfully loaded YOLO models '{POSE_MODEL_NAME}' and '{OBJECT_MODEL_NAME}' globally.")
except Exception as e:
    logging.error(f"Error loading YOLO models globally: {e}")
    pose_model = None
    object_model = None

# Define keypoint connections for drawing skeleton (COCO format)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Torso/Arms
    (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16) # Legs
]
KEYPOINT_COLOR = (0, 255, 0) # Green
SKELETON_COLOR = (255, 0, 0) # Blue
RACKET_COLOR = (255, 255, 0) # Yellow
CONFIDENCE_THRESHOLD = 0.5 # Minimum confidence to draw a keypoint/skeleton
RACKET_CONFIDENCE_THRESHOLD = 0.3 # Lower threshold for racket detection

def draw_skeleton(frame, keypoints, connections, kp_color, sk_color, threshold, model_shape):
    """Draws keypoints and skeleton on the frame."""
    img_h, img_w = frame.shape[:2]
    keypoints_data = keypoints.data.cpu().numpy() # Get keypoints as numpy array

    for person_kpts in keypoints_data: # Iterate through detected persons
        valid_kpts = {}
        # Draw Keypoints
        for i, kpt in enumerate(person_kpts):
            x, y, conf = kpt
            if conf >= threshold:
                x, y = int(x), int(y) # Use direct coordinates if results are already scaled
                cv2.circle(frame, (x, y), 5, kp_color, -1)
                valid_kpts[i] = (x, y) # Store valid keypoints for skeleton

        # Draw Skeleton
        for i, j in connections:
            if i in valid_kpts and j in valid_kpts:
                pt1 = valid_kpts[i]
                pt2 = valid_kpts[j]
                cv2.line(frame, pt1, pt2, sk_color, 2)
    return frame

def draw_racket(frame, detections, color, threshold):
    """Draws tennis racket bounding boxes and keypoints on the frame."""
    for detection in detections:
        # Get the class name and confidence
        class_name = detection.names[int(detection.boxes.cls[0])]
        confidence = float(detection.boxes.conf[0])
        
        # Only process if it's a tennis racket with sufficient confidence
        if class_name.lower() == "tennis racket" and confidence >= threshold:
            # Get the bounding box coordinates
            box = detection.boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw keypoints if available
            if hasattr(detection, 'keypoints') and detection.keypoints is not None:
                keypoints = detection.keypoints.data.cpu().numpy()[0]
                for kp in keypoints:
                    x, y, conf = kp
                    if conf >= threshold:
                        x, y = int(x), int(y)
                        cv2.circle(frame, (x, y), 5, color, -1)
            
            # Add confidence label
            label = f"Racket {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def extract_skeleton_keypoints(keypoints) -> List[Dict[str, Any]]:
    """Extract skeleton keypoints from YOLO results and convert to JSON-serializable format."""
    keypoints_data = keypoints.data.cpu().numpy()
    frame_keypoints = []
    
    for person_kpts in keypoints_data:
        person_keypoints = []
        for i, kpt in enumerate(person_kpts):
            x, y, conf = kpt
            if conf >= CONFIDENCE_THRESHOLD:
                person_keypoints.append({
                    "keypoint_id": i,
                    "x": float(x),
                    "y": float(y),
                    "confidence": float(conf)
                })
        if person_keypoints:  # Only add if we found valid keypoints
            frame_keypoints.append(person_keypoints)
    
    return frame_keypoints

def extract_racket_keypoints(detections) -> List[Dict[str, Any]]:
    """Extract tennis racket keypoints from YOLO results and convert to JSON-serializable format."""
    frame_rackets = []
    
    for detection in detections:
        class_name = detection.names[int(detection.boxes.cls[0])]
        confidence = float(detection.boxes.conf[0])
        
        if class_name.lower() == "tennis racket" and confidence >= RACKET_CONFIDENCE_THRESHOLD:
            racket_data = {
                "confidence": confidence,
                "bbox": detection.boxes.xyxy[0].cpu().numpy().tolist(),
                "keypoints": []
            }
            
            if hasattr(detection, 'keypoints') and detection.keypoints is not None:
                keypoints = detection.keypoints.data.cpu().numpy()[0]
                for kp in keypoints:
                    x, y, conf = kp
                    if conf >= RACKET_CONFIDENCE_THRESHOLD:
                        racket_data["keypoints"].append({
                            "x": float(x),
                            "y": float(y),
                            "confidence": float(conf)
                        })
            
            frame_rackets.append(racket_data)
    
    return frame_rackets

def extract_keypoints_for_avatar(keypoints) -> List[Dict[str, Any]]:
    """
    Extract keypoints from YOLO results and convert to the format expected by Avatar.tsx.
    
    Avatar.tsx expects:
    {
      "frames": [
        {
          "keypoints": [
            { "x": 123.45, "y": 67.89, "confidence": 0.95 },
            ...
          ]
        },
        ...
      ]
    }
    """
    keypoints_data = keypoints.data.cpu().numpy()
    avatar_keypoints = []
    
    # We'll use the first person detected in each frame
    if len(keypoints_data) > 0:
        person_kpts = keypoints_data[0]  # Take the first person
        for i, kpt in enumerate(person_kpts):
            x, y, conf = kpt
            if conf >= CONFIDENCE_THRESHOLD:
                avatar_keypoints.append({
                    "x": float(x),
                    "y": float(y),
                    "confidence": float(conf)
                })
    
    return avatar_keypoints

def process_video(input_file: str, output_dir: str) -> str:
    """
    Process a video file with YOLO v8 pose detection and overlay keypoints.
    The output video filename is constructed as {input_name}-skeleton-racket{extension}.
    Also saves keypoints to JSON files: {name}-skeleton.json, {name}-racket.json, and {name}-avatar.json
    
    Args:
        input_file (str): Path to the input video file.
        output_dir (str): Path to the output directory.
        
    Returns:
        str: The full path to the processed output video file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the file basename and extension
    basename = os.path.basename(input_file)
    name, ext = os.path.splitext(basename)
    output_file = os.path.join(output_dir, f"{name}-skeleton-racket{ext}")
    skeleton_json = os.path.join(output_dir, f"{name}-skeleton.json")
    racket_json = os.path.join(output_dir, f"{name}-racket.json")
    avatar_json = os.path.join(output_dir, f"{name}-avatar.json")

    logging.info(f"Input file: {input_file}")
    logging.info(f"Output video: {output_file}")
    logging.info(f"Output skeleton JSON: {skeleton_json}")
    logging.info(f"Output racket JSON: {racket_json}")
    logging.info(f"Output avatar JSON: {avatar_json}")

    # Initialize lists to store all frame keypoints
    all_skeleton_keypoints = []
    all_racket_keypoints = []
    all_avatar_keypoints = []  # New list for Avatar.tsx format

    # Open the input video
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        logging.error("Cannot open the input video file.")
        return ""

    # Check video rotation metadata
    rotation = 0
    try:
        # Try to get rotation metadata (works on some platforms)
        rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        logging.info(f"Video rotation metadata: {rotation} degrees")
    except:
        logging.info("Could not get video rotation metadata, assuming 0 degrees")
    
    # Retrieve video properties (fps, width, height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Determine output dimensions based on rotation
    if rotation in [90, 270]:
        # Swap width and height for 90/270 degree rotation
        out_width, out_height = height, width
    else:
        out_width, out_height = width, height
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (out_width, out_height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply rotation if needed
        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Run YOLO v8 pose detection on the frame
        pose_results = pose_model(frame)[0]
        
        # Run YOLO v8 object detection on the frame
        object_results = object_model(frame)[0]
        
        # Extract and store keypoints
        if hasattr(pose_results, "keypoints"):
            # Original format for skeleton visualization
            frame_skeleton_keypoints = extract_skeleton_keypoints(pose_results.keypoints)
            all_skeleton_keypoints.append({
                "frame": frame_count,
                "keypoints": frame_skeleton_keypoints
            })
            
            # New format for Avatar.tsx
            frame_avatar_keypoints = extract_keypoints_for_avatar(pose_results.keypoints)
            all_avatar_keypoints.append({
                "keypoints": frame_avatar_keypoints
            })
            
            frame = draw_skeleton(frame, pose_results.keypoints, SKELETON_CONNECTIONS, 
                                KEYPOINT_COLOR, SKELETON_COLOR, CONFIDENCE_THRESHOLD, 
                                frame.shape[:2])
        
        # Extract and store racket keypoints
        frame_racket_keypoints = extract_racket_keypoints(object_results)
        all_racket_keypoints.append({
            "frame": frame_count,
            "rackets": frame_racket_keypoints
        })
        frame = draw_racket(frame, object_results, RACKET_COLOR, RACKET_CONFIDENCE_THRESHOLD)
        
        # Write the processed frame to the output video
        out.write(frame)
        frame_count += 1
        if frame_count % 100 == 0: # Log progress periodically
            logging.info(f"Processed {frame_count} frames...")
    
    # Save keypoints to JSON files
    with open(skeleton_json, 'w') as f:
        json.dump(all_skeleton_keypoints, f, indent=2)
    with open(racket_json, 'w') as f:
        json.dump(all_racket_keypoints, f, indent=2)
    
    # Save keypoints in Avatar.tsx format
    with open(avatar_json, 'w') as f:
        json.dump({"frames": all_avatar_keypoints}, f, indent=2)
    
    # Cleanup
    cap.release()
    out.release()
    logging.info(f"Video processing complete. Total frames processed: {frame_count}")
    logging.info(f"Keypoints saved to {skeleton_json}, {racket_json}, and {avatar_json}")
    
    return output_file

def main():
    logging.info("Starting pose tracking...")
    # Configure basic logging format
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Process a video for pose detection with YOLO v8.")
    parser.add_argument("input_file", help="Path to the input video file")
    parser.add_argument("output_dir", help="Path to the output directory")
    
    args = parser.parse_args()
    
    output_filepath = process_video(args.input_file, args.output_dir)
    
    if output_filepath:
        logging.info(f"Processed video saved to: {output_filepath}")
    else:
        logging.error("Processing failed.")

if __name__ == "__main__":
    main()
