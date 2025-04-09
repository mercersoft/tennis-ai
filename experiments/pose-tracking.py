import os
import cv2
import logging
import argparse
from ultralytics import YOLO

def process_video(input_file: str, output_dir: str) -> str:
    """
    Process a video file with YOLO v8 pose detection and overlay keypoints.
    The output video filename is constructed as {input_name}-output{extension}.
    
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
    output_file = os.path.join(output_dir, f"{name}-output{ext}")

    logging.info(f"Input file: {input_file}")
    logging.info(f"Output file: {output_file}")

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

    # Initialize the YOLO v8 pose model.
    # Ensure that the model file ("yolov8n-pose.pt") is located in your working directory
    model = YOLO("yolov8n-pose.pt")
    
    logging.info("Processing video frames...")
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
        results = model(frame)[0]
        
        # If keypoints are detected, overlay them.
        if hasattr(results, "keypoints"):
            for kp in results.keypoints.data.cpu().numpy():
                for point in kp:
                    x, y, conf = point
                    # Draw keypoints only if the confidence is above a threshold (e.g., 0.5)
                    if conf > 0.5:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        
        # Write the processed frame to the output video
        out.write(frame)
    
    # Cleanup
    cap.release()
    out.release()
    logging.info("Video processing complete.")
    
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
