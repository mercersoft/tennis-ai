import json
import math

def generate_keypoints(frame_count=100):
    frames = []
    
    # Base keypoints in T-pose
    base_keypoints = [
        {"x": 0, "y": 0, "confidence": 0.9},      # nose
        {"x": 0, "y": -0.2, "confidence": 0.9},   # neck
        {"x": -0.5, "y": -0.2, "confidence": 0.9},# left shoulder
        {"x": -1.0, "y": -0.2, "confidence": 0.9},# left elbow
        {"x": -1.5, "y": -0.2, "confidence": 0.9},# left wrist
        {"x": 0.5, "y": -0.2, "confidence": 0.9}, # right shoulder
        {"x": 1.0, "y": -0.2, "confidence": 0.9}, # right elbow
        {"x": 1.5, "y": -0.2, "confidence": 0.9}, # right wrist
        {"x": -0.2, "y": -1.0, "confidence": 0.9},# left hip
        {"x": 0.2, "y": -1.0, "confidence": 0.9}, # right hip
        {"x": -0.3, "y": -1.8, "confidence": 0.9},# left knee
        {"x": 0.3, "y": -1.8, "confidence": 0.9}, # right knee
        {"x": -0.3, "y": -2.5, "confidence": 0.9},# left ankle
        {"x": 0.3, "y": -2.5, "confidence": 0.9}  # right ankle
    ]
    
    for frame in range(frame_count):
        # Copy base keypoints
        frame_keypoints = base_keypoints.copy()
        
        # Calculate right arm position
        # Convert -1 to 1 range to our desired angle range
        # -45 degrees = -pi/4, 60 degrees = pi/3
        base_angle = math.sin(frame * 2 * math.pi / frame_count)
        # Map -1 to 1 to our desired range
        if base_angle > 0:
            angle = (base_angle * math.pi/3)  # Scale up to 60 degrees
        else:
            angle = (base_angle * math.pi/4)  # Scale down to 45 degrees
        
        # Update right arm keypoints (shoulder, elbow, wrist)
        # Right shoulder (minimal movement)
        frame_keypoints[5] = {
            "x": 0.5,
            "y": -0.2 + math.sin(angle) * 0.1,
            "confidence": 0.9
        }
        
        # Right elbow
        frame_keypoints[6] = {
            "x": 0.5 + math.cos(angle) * 0.5,
            "y": -0.2 + math.sin(angle) * 0.5,
            "confidence": 0.9
        }
        
        # Right wrist
        frame_keypoints[7] = {
            "x": 0.5 + math.cos(angle) * 1.0,
            "y": -0.2 + math.sin(angle) * 1.0,
            "confidence": 0.9
        }
        
        frames.append({"keypoints": frame_keypoints})
    
    return {"frames": frames}

# Generate and save the keypoints
data = generate_keypoints(100)
with open('website/src/assets/test.json', 'w') as f:
    json.dump(data, f, indent=2) 