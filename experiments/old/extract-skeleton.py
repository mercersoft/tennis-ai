import logging
import cv2
import numpy as np
from ultralytics import YOLO
import json
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_video(input_path, output_path):
    # Load the pose estimation model
    pose_model = YOLO('yolov8n-pose.pt')
    logger.info("Loaded pose estimation model")

    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error("Error: Could not open video file")
        return

    frames_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run pose detection
        results = pose_model(frame, verbose=False)
        
        if len(results) > 0 and len(results[0].keypoints) > 0:
            # Get keypoints for the first person detected
            keypoints = results[0].keypoints.data[0].cpu().numpy()
            
            # Extract x, y coordinates and confidence scores
            frame_keypoints = []
            for kp in keypoints:
                # Normalize coordinates to [-1, 1] range
                x = (float(kp[0]) / frame.shape[1] * 2) - 1
                y = -((float(kp[1]) / frame.shape[0] * 2) - 1)  # Flip Y axis for 3D space
                confidence = float(kp[2])
                
                frame_keypoints.append({
                    "x": float(x),
                    "y": float(y),
                    "confidence": confidence
                })
            
            frames_data.append({"keypoints": frame_keypoints})
            
            if frame_count % 30 == 0:
                logger.info(f"Processed frame {frame_count}")
        
        frame_count += 1

    cap.release()

    # Save the keypoints data
    output_data = {"frames": frames_data}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Processed {frame_count} frames")
    logger.info(f"Saved keypoints data to {output_path}")

def main():
    if len(sys.argv) != 2:
        logger.error("Usage: python extract-skeleton.py <input_video>")
        sys.exit(1)

    input_video = sys.argv[1]
    output_path = "../website/src/assets/skeleton.json"

    logger.info(f"Processing video: {input_video}")
    process_video(input_video, output_path)

if __name__ == "__main__":
    main()
