import json
import math

def generate_keypoints(frame_num, total_frames=100):
    # Wave animation parameters
    wave_amplitude = 40  # How far the arms move
    wave_frequency = 2   # How many complete waves in the animation
    
    # Calculate the phase of the wave for this frame
    phase = (frame_num / total_frames) * 2 * math.pi * wave_frequency
    
    # Base positions for a T-pose
    base_keypoints = [
        # Head
        {"x": 320, "y": 200, "confidence": 0.9},  # 0: Nose
        {"x": 310, "y": 190, "confidence": 0.9},  # 1: Left Eye
        {"x": 330, "y": 190, "confidence": 0.9},  # 2: Right Eye
        {"x": 300, "y": 195, "confidence": 0.9},  # 3: Left Ear
        {"x": 340, "y": 195, "confidence": 0.9},  # 4: Right Ear
        
        # Shoulders and Arms
        {"x": 280, "y": 240, "confidence": 0.9},  # 5: Left Shoulder
        {"x": 360, "y": 240, "confidence": 0.9},  # 6: Right Shoulder
        {"x": 260, "y": 300, "confidence": 0.9},  # 7: Left Elbow
        {"x": 380, "y": 300, "confidence": 0.9},  # 8: Right Elbow
        {"x": 240, "y": 360, "confidence": 0.9},  # 9: Left Wrist
        {"x": 400, "y": 360, "confidence": 0.9},  # 10: Right Wrist
        
        # Hips and Legs
        {"x": 300, "y": 320, "confidence": 0.9},  # 11: Left Hip
        {"x": 340, "y": 320, "confidence": 0.9},  # 12: Right Hip
        {"x": 290, "y": 400, "confidence": 0.9},  # 13: Left Knee
        {"x": 350, "y": 400, "confidence": 0.9},  # 14: Right Knee
        {"x": 280, "y": 460, "confidence": 0.9},  # 15: Left Ankle
        {"x": 360, "y": 460, "confidence": 0.9}   # 16: Right Ankle
    ]
    
    # Animate the arms (shoulders, elbows, and wrists)
    arm_indices = [5, 6, 7, 8, 9, 10]  # Indices for arm keypoints
    
    for idx in arm_indices:
        # Left side moves up while right side moves down and vice versa
        offset = wave_amplitude * math.sin(phase)
        if idx in [5, 7, 9]:  # Left arm
            base_keypoints[idx]["y"] += offset
        else:  # Right arm
            base_keypoints[idx]["y"] -= offset
            
    # Add a slight head movement
    head_offset = wave_amplitude * 0.2 * math.sin(phase * 0.5)
    for idx in range(5):  # Head keypoints
        base_keypoints[idx]["y"] += head_offset
    
    return base_keypoints

# Generate 100 frames of animation
frames = []
for i in range(100):
    frames.append({
        "keypoints": generate_keypoints(i)
    })

# Save to JSON file
output = {"frames": frames}
with open("website/src/assets/test.json", "w") as f:
    json.dump(output, f, indent=2) 