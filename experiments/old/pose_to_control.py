import json
import math
import numpy as np
import argparse
import warnings
from collections import defaultdict
import sys # Import sys for exiting

# --- Configuration ---
# Keypoint names used in the JSON input (MUST MATCH YOUR JSON FILE!)
KP_NAMES = {
    "nose": "Nose",
    "l_shoulder": "L_Shoulder",
    "r_shoulder": "R_Shoulder",
    "l_hip": "L_Hip",
    "r_hip": "R_Hip",
    "l_ankle": "L_Ankle",
    "r_ankle": "R_Ankle",
}

# Minimum confidence score to consider a keypoint valid
MIN_CONFIDENCE = 0.3

# Smoothing factor (0 = no smoothing, closer to 1 = more smoothing)
SMOOTHING_FACTOR_YAW = 0.6
SMOOTHING_FACTOR_SCALE = 0.6

# --- Average Anthropometric Data (in Meters) ---
# (Same as before)
ANTHRO_DATA = {
    "male": {
        "shoulder_width": 0.41,
        "hip_width": 0.33,
        "torso_height_v": 0.52
    },
    "female": {
        "shoulder_width": 0.38,
        "hip_width": 0.35,
        "torso_height_v": 0.48
    }
}

# --- Helper Functions (reuse from previous script) ---
# get_keypoint, calculate_distance, calculate_midpoint, smooth_values
# (Copied from previous version - no changes needed here)
def get_keypoint(person_data, kp_name):
    """Safely retrieves keypoint data [x, y, conf]"""
    if kp_name in person_data["keypoints"]:
        kp = person_data["keypoints"][kp_name]
        if isinstance(kp, (list, tuple)) and len(kp) == 3:
             if kp[2] >= MIN_CONFIDENCE:
                 return np.array(kp[:2]) # Return [x, y] as numpy array
             else:
                 return None
        else:
             warnings.warn(f"Invalid format for keypoint {kp_name}: {kp}")
             return None
    return None

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points"""
    if p1 is None or p2 is None:
        return None
    return np.linalg.norm(p1 - p2)

def calculate_midpoint(p1, p2):
    """Calculates the midpoint between two points"""
    if p1 is None or p2 is None:
        return None
    return (p1 + p2) / 2.0

def smooth_values(values, alpha):
    """Applies exponential moving average smoothing."""
    if alpha is None or not (0 < alpha < 1):
        return values # No smoothing
    smoothed = []
    last_val = None
    num_valid_in = sum(1 for v in values if v is not None)
    if num_valid_in == 0: return values # Avoid issues if all are None

    first_valid_idx = -1
    for i, v in enumerate(values):
        if v is not None:
            first_valid_idx = i
            break

    if first_valid_idx != -1:
        last_val = values[first_valid_idx]
        for i in range(len(values)):
            val = values[i]
            if val is None:
                smoothed.append(None)
                continue

            if last_val is None:
                smoothed_val = val
            else:
                smoothed_val = alpha * last_val + (1 - alpha) * val
            smoothed.append(smoothed_val)
            last_val = smoothed_val
    else:
        smoothed = values
    return smoothed


# --- NEW: Function to Check Keypoint Names ---
def check_keypoint_names(all_frames_data, expected_kp_name_map):
    """
    Checks the first valid frame for expected keypoint names and structure.

    Args:
        all_frames_data (list): The loaded JSON data (list of frames).
        expected_kp_name_map (dict): The script's KP_NAMES dictionary.

    Returns:
        bool: True if essential keypoints seem present, False otherwise.
    """
    print("\nChecking JSON structure and keypoint names...")
    first_valid_person_keypoints = None
    first_valid_frame_id = "N/A"

    for frame_data in all_frames_data:
        frame_id = frame_data.get("frame_id", "N/A")
        persons = frame_data.get("persons")
        if persons and isinstance(persons, list) and len(persons) > 0:
            person = persons[0]
            if isinstance(person, dict) and "keypoints" in person and isinstance(person["keypoints"], dict):
                first_valid_person_keypoints = person["keypoints"]
                first_valid_frame_id = frame_id
                break # Found the first valid structure

    if first_valid_person_keypoints is None:
        print("\nError: Could not find the expected JSON structure in any frame.")
        print("Expected: A list of frames, each frame object having a 'persons' list.")
        print("          The first item in 'persons' should be a dictionary with a 'keypoints' dictionary inside it.")
        print("Please verify the structure of your input JSON file.")
        return False # Indicate failure

    print(f"Found 'keypoints' dictionary in first valid frame (ID: {first_valid_frame_id}). Analyzing names...")

    actual_keys = set(first_valid_person_keypoints.keys())
    expected_names = set(expected_kp_name_map.values()) # Get the names like "L_Shoulder"

    # Define essential keys needed for Pass 1 (height calculation)
    essential_pass1_keys = {
        expected_kp_name_map["l_shoulder"], expected_kp_name_map["r_shoulder"],
        expected_kp_name_map["l_hip"], expected_kp_name_map["r_hip"],
        expected_kp_name_map["l_ankle"], expected_kp_name_map["r_ankle"]
    }

    missing_keys = expected_names - actual_keys
    found_keys = expected_names.intersection(actual_keys)
    extra_keys = actual_keys - expected_names
    missing_essential_keys = essential_pass1_keys - actual_keys

    print(f"\n--- Keypoint Name Analysis (Expected vs. Actual in Frame {first_valid_frame_id}) ---")
    print(f"Expected Key Names (from script's KP_NAMES): {len(expected_names)}")
    print(f"Actual Key Names Found in JSON: {len(actual_keys)}")
    print(f"  - Found Matching Expected Keys: {len(found_keys)}")
    if found_keys: print(f"    {sorted(list(found_keys))}")
    print(f"  - Expected Keys NOT Found in JSON: {len(missing_keys)}")
    if missing_keys: print(f"    {sorted(list(missing_keys))}")
    print(f"  - Unexpected/Extra Keys Found in JSON (Not in script's KP_NAMES): {len(extra_keys)}")
    if extra_keys: print(f"    {sorted(list(extra_keys))}")
    print("--- End Analysis ---")

    if missing_essential_keys:
        print("\n*** Error: Essential keypoints required for height/scale calculation are MISSING in the JSON data! ***")
        print(f"The following essential keys were not found: {sorted(list(missing_essential_keys))}")
        print("This means the script cannot reliably calculate the person's size.")
        print("\nPlease check:")
        print("  1. If the keypoint names in your JSON file (listed under 'Actual Key Names Found'/'Unexpected/Extra Keys')")
        print("     EXACTLY match the names expected by the script (listed under 'Expected Keys NOT Found').")
        print("     (Comparison is case-sensitive! 'L_Shoulder' != 'l_shoulder').")
        print("  2. If your pose estimation process (YOLO Pose) actually generated these keypoints.")
        print("\n--> If names mismatch, UPDATE the `KP_NAMES` dictionary at the top of this script and re-run.")
        return False # Indicate failure
    else:
        print("\n✅ Essential keypoint names required for height calculation (Shoulders, Hips, Ankles) seem to be present.")
        if missing_keys:
            # Check if nose is missing, as it's useful but not strictly essential for this check
            if expected_kp_name_map.get("nose") in missing_keys:
                 print("   (Note: 'Nose' keypoint is missing or doesn't match. Orientation calculation might be less accurate.)")
            else:
                 print(f"   (Note: Some non-essential expected keys might be missing: {sorted(list(missing_keys))})")
        if extra_keys:
            print("   (Note: Your JSON contains extra keypoint names not used by this script.)")
        print("Keypoint name check passed. Continuing...")
        return True # Indicate success


# --- Main Processing Function ---

def process_pose_data(input_json_path, output_json_path, gender):
    """
    Processes YOLO Pose JSON to extract avatar control parameters using anthropometric data.
    Args:
        input_json_path (str): Path to the input JSON file.
        output_json_path (str): Path to save the output control JSON file.
        gender (str): 'male' or 'female' for anthropometric data selection.
    """
    if gender not in ANTHRO_DATA:
        print(f"Error: Invalid gender '{gender}'. Please use 'male' or 'female'.")
        return # Exit if gender is invalid

    avg_data = ANTHRO_DATA[gender]
    print(f"Using average anthropometric data for gender: {gender}")
    # Print avg data... (same as before)

    try:
        with open(input_json_path, 'r') as f:
            all_frames_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_json_path}")
        return

    if not isinstance(all_frames_data, list):
         print("Error: Expected input JSON to be a list of frames.")
         return

    print(f"Loaded {len(all_frames_data)} frames from {input_json_path}")

    # *** Run the Keypoint Name Check ***
    if not check_keypoint_names(all_frames_data, KP_NAMES):
        sys.exit(1) # Exit script if check fails

    # --- Pass 1: Cache Keypoints and Calculate Average Projected Body Height ---
    # (Same Pass 1 logic as the previous version)
    keypoints_cache = {}
    projected_body_heights = []
    valid_frame_count_pass1 = 0

    print("\nStarting Pass 1: Caching keypoints and calculating avg body height...")
    # (Loop through frames, get keypoints using get_keypoint with MIN_CONFIDENCE,
    # calculate body_height_proj, store in projected_body_heights...)
    # ... (rest of Pass 1 logic is identical to previous version) ...
    for frame_data in all_frames_data:
        frame_id = frame_data.get("frame_id")
        persons = frame_data.get("persons")

        if frame_id is None: # Allow frames without persons for caching None
             keypoints_cache[frame_id] = None
             continue

        if not persons: # Handle frames where no person was detected
            keypoints_cache[frame_id] = None
            continue

        person = persons[0] # Use the first person detected
        kps = {}
        valid_person = True
        # Require core points for basic processing and height calculation
        required_kps_pass1 = [
            KP_NAMES["l_shoulder"], KP_NAMES["r_shoulder"],
            KP_NAMES["l_hip"], KP_NAMES["r_hip"],
            KP_NAMES["l_ankle"], KP_NAMES["r_ankle"]
        ]
        for name in required_kps_pass1:
            kp_data = get_keypoint(person, name) # get_keypoint applies MIN_CONFIDENCE
            if kp_data is None:
                valid_person = False
                # No need to break, capture all available points below
            kps[name] = kp_data # Store kp_data (might be None)

        # Also try to get nose if available for orientation refinement
        kps[KP_NAMES["nose"]] = get_keypoint(person, KP_NAMES["nose"])

        keypoints_cache[frame_id] = kps # Store collected keypoints (some might be None)

        # Check validity specifically for height calculation *after* caching
        if all(kps.get(name) is not None for name in required_kps_pass1):
            mid_shoulder = calculate_midpoint(kps[KP_NAMES["l_shoulder"]], kps[KP_NAMES["r_shoulder"]])
            mid_ankle = calculate_midpoint(kps[KP_NAMES["l_ankle"]], kps[KP_NAMES["r_ankle"]])
            if mid_shoulder is not None and mid_ankle is not None:
                body_height_proj = abs(mid_shoulder[1] - mid_ankle[1]) # Vertical distance
                if body_height_proj > 1e-6: # Use a small threshold instead of > 0
                    projected_body_heights.append(body_height_proj)
                    valid_frame_count_pass1 += 1

    if not projected_body_heights:
        # This error should be less likely now if the name check passed,
        # meaning the problem is likely low confidence or missing points in *all* frames.
        print("\nError: Could not calculate projected body height from any frame.")
        print(f"       Although essential keypoint *names* seem correct, no single frame had")
        print(f"       all required keypoints (Shoulders, Hips, Ankles) simultaneously valid")
        print(f"       with confidence >= {MIN_CONFIDENCE}.")
        print(f"       Consider lowering the confidence threshold ('-c' option).")
        print("Cannot determine reference scale.")
        return # Exit if still no valid heights found

    average_projected_body_height = np.mean(projected_body_heights)
    print(f"\nPass 1 complete. Processed {valid_frame_count_pass1} valid frames for height.")
    print(f"  Average Projected Body Height (Shoulder-Ankle): {average_projected_body_height:.2f} pixels")


    # --- Pass 2: Calculate Yaw and Scale per Frame ---
    # (Same Pass 2 logic as previous version)
    # ... loop through frames, use cached keypoints ...
    # ... calculate V_torso_proj, pixels_per_meter, w_sh_max_expected ...
    # ... calculate w_sh_proj, cos_theta_sh, angle_mag ...
    # ... determine is_front_view, turn_bias, yaw ...
    # ... calculate current_body_height_proj, scale ...
    # ... store results in results_raw ...
    results_raw = []
    print("\nStarting Pass 2: Calculating parameters per frame...")
    avg_torso_h_real = avg_data['torso_height_v']
    avg_shoulder_w_real = avg_data['shoulder_width']
    avg_hip_w_real = avg_data['hip_width']

    for frame_data in all_frames_data:
        frame_id = frame_data.get("frame_id")
        # Handle cases where frame_id might be missing in source JSON
        if frame_id is None:
             warnings.warn("Found frame data without 'frame_id'. Skipping.")
             continue

        kps = keypoints_cache.get(frame_id) # Get cached keypoints

        output_frame = {"frame_id": frame_id, "controls": {"yaw": None, "scale": None}}

        # Check if essential keypoints for *this frame's* calculation are present
        # Need shoulders and hips for yaw scale estimation. Need ankles for body height scale.
        required_for_yaw = [KP_NAMES["l_shoulder"], KP_NAMES["r_shoulder"], KP_NAMES["l_hip"], KP_NAMES["r_hip"]]
        required_for_scale = [KP_NAMES["l_shoulder"], KP_NAMES["r_shoulder"], KP_NAMES["l_ankle"], KP_NAMES["r_ankle"]] # Need shoulders for mid_shoulder

        if kps is None or not all(kps.get(name) is not None for name in required_for_yaw):
            # Cannot calculate yaw if shoulders/hips invalid for this frame
            if kps is not None and all(kps.get(name) is not None for name in required_for_scale):
                 # Can still calculate scale
                 mid_shoulder = calculate_midpoint(kps[KP_NAMES["l_shoulder"]], kps[KP_NAMES["r_shoulder"]])
                 mid_ankle = calculate_midpoint(kps[KP_NAMES["l_ankle"]], kps[KP_NAMES["r_ankle"]])
                 current_body_height_proj = abs(mid_shoulder[1] - mid_ankle[1])
                 if current_body_height_proj > 1e-6 and average_projected_body_height > 1e-6:
                      output_frame["controls"]["scale"] = current_body_height_proj / average_projected_body_height
                 else:
                      output_frame["controls"]["scale"] = 1.0 # Default
            else:
                 # Cannot calculate scale either
                 pass # Keep scale as None if required points missing
            results_raw.append(output_frame) # Add frame with potentially None controls
            continue # Skip rest of loop for this frame


        # --- Extract Keypoints (now guaranteed non-None for yaw calc) ---
        l_sh, r_sh = kps[KP_NAMES["l_shoulder"]], kps[KP_NAMES["r_shoulder"]]
        l_hip, r_hip = kps[KP_NAMES["l_hip"]], kps[KP_NAMES["r_hip"]]
        nose = kps.get(KP_NAMES["nose"]) # Optional

        # --- Calculate Midpoints ---
        mid_shoulder = calculate_midpoint(l_sh, r_sh)
        mid_hip = calculate_midpoint(l_hip, r_hip)
        # mid_ankle needed later for scale

        # --- 1. Calculate Yaw ---
        # (Rest of Yaw calculation logic is identical to previous version)
        # ... estimate pixels_per_meter using v_torso_proj ...
        # ... calculate w_sh_max_expected, w_hi_max_expected ...
        # ... calculate w_sh_proj, w_hi_proj ...
        # ... calculate cos_theta_sh, cos_theta_hi, angle_mag ...
        # ... determine is_front_view, turn_bias, yaw ...
        yaw = None
        pixels_per_meter = None
        w_sh_max_expected = None
        w_hi_max_expected = None

        v_torso_proj = abs(mid_shoulder[1] - mid_hip[1])
        if v_torso_proj > 1e-6 and avg_torso_h_real > 1e-6:
            pixels_per_meter = v_torso_proj / avg_torso_h_real
            w_sh_max_expected = avg_shoulder_w_real * pixels_per_meter
            w_hi_max_expected = avg_hip_w_real * pixels_per_meter
        # else: pixels_per_meter remains None, handled below

        if pixels_per_meter is not None:
             w_sh_proj = abs(l_sh[0] - r_sh[0])
             w_hi_proj = abs(l_hip[0] - r_hip[0])

             cos_theta_sh = 0.0
             if w_sh_max_expected > 1e-6:
                 cos_theta_sh = np.clip(w_sh_proj / w_sh_max_expected, 0.0, 1.0)

             cos_theta_hi = 0.0
             if w_hi_max_expected > 1e-6:
                 cos_theta_hi = np.clip(w_hi_proj / w_hi_max_expected, 0.0, 1.0)

             # If widths were validly computed
             if w_sh_max_expected > 1e-6 and w_hi_max_expected > 1e-6:
                  angle_mag_sh = math.acos(cos_theta_sh)
                  angle_mag_hi = math.acos(cos_theta_hi)
                  angle_mag = (angle_mag_sh + angle_mag_hi) / 2.0

                  is_front_view = (l_sh[0] < r_sh[0])
                  turn_bias = 0.0
                  if nose is not None:
                      face_dir_x = nose[0] - mid_shoulder[0]
                      if w_sh_proj > 1: turn_bias = face_dir_x / w_sh_proj

                  side_on_threshold = math.radians(80)
                  # (Combine magnitude and sign/bias - same logic as before)
                  if is_front_view:
                      if angle_mag < side_on_threshold:
                          yaw = math.copysign(angle_mag, turn_bias if abs(turn_bias) > 0.1 else 1.0)
                          if angle_mag < math.radians(5): yaw = 0.0
                      else: yaw = math.pi/2 if turn_bias >= 0 else -math.pi/2
                  else: # Back view
                      if angle_mag < side_on_threshold:
                          angle_offset = math.copysign(angle_mag, -turn_bias if abs(turn_bias) > 0.1 else 1.0)
                          yaw = math.pi + angle_offset
                          if yaw > math.pi: yaw -= 2 * math.pi
                          if yaw <= -math.pi: yaw += 2 * math.pi
                          if angle_mag < math.radians(5): yaw = math.pi if yaw > 0 else -math.pi
                      else: yaw = math.pi/2 if turn_bias >= 0 else -math.pi/2


        # --- 2. Calculate Scale ---
        scale = None
        # Check if ankle keypoints are valid for this frame
        l_ank = kps.get(KP_NAMES["l_ankle"])
        r_ank = kps.get(KP_NAMES["r_ankle"])
        if l_ank is not None and r_ank is not None:
             mid_ankle = calculate_midpoint(l_ank, r_ank)
             if mid_ankle is not None:
                 current_body_height_proj = abs(mid_shoulder[1] - mid_ankle[1]) # mid_shoulder is valid here
                 if current_body_height_proj > 1e-6 and average_projected_body_height > 1e-6:
                     scale = current_body_height_proj / average_projected_body_height
                 else: scale = 1.0 # Default scale
        # If ankles invalid, scale remains None unless defaulted above

        output_frame["controls"]["yaw"] = yaw
        output_frame["controls"]["scale"] = scale if scale is not None else 1.0 # Ensure scale isn't None in output
        results_raw.append(output_frame)


    # --- Pass 3: Apply Smoothing ---
    # (Same Pass 3 logic as previous version)
    # ... extract raw_yaws, raw_scales ...
    # ... apply smooth_values ...
    # ... combine into final_results ...
    print("\nStarting Pass 3: Smoothing...")
    raw_yaws = [f["controls"]["yaw"] for f in results_raw]
    raw_scales = [f["controls"]["scale"] for f in results_raw]

    smoothed_yaws = smooth_values(raw_yaws, SMOOTHING_FACTOR_YAW)
    smoothed_scales = smooth_values(raw_scales, SMOOTHING_FACTOR_SCALE)

    final_results = []
    for i, frame_data in enumerate(results_raw):
        final_results.append({
            "frame_id": frame_data["frame_id"],
            "controls": {
                "yaw": smoothed_yaws[i],
                # Ensure scale output isn't None, default to 1.0 if smoothing fails/input was None
                "scale": smoothed_scales[i] if smoothed_scales[i] is not None else 1.0
            }
        })

    # --- Save Output ---
    # (Same save logic as previous version)
    try:
        with open(output_json_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"\nSuccessfully saved control parameters to {output_json_path}")
    except IOError:
        print(f"\nError: Could not write output file to {output_json_path}")

# --- Command Line Argument Parsing ---
# (Same as previous version)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate avatar control parameters (Yaw, Scale) from YOLO Pose JSON using anthropometric data.")
    parser.add_argument("input_json", help="Path to the input JSON file from YOLO Pose.")
    parser.add_argument("output_json", help="Path to save the output control parameters JSON file.")
    parser.add_argument("--gender", required=True, choices=['male', 'female'],
                        help="Gender of the person in the video ('male' or 'female').")
    parser.add_argument("-c", "--confidence", type=float, default=MIN_CONFIDENCE,
                        help=f"Minimum keypoint confidence threshold (default: {MIN_CONFIDENCE})")
    parser.add_argument("--smooth_yaw", type=float, default=SMOOTHING_FACTOR_YAW,
                        help=f"Exponential smoothing factor for yaw (0-1, 0 or None to disable, default: {SMOOTHING_FACTOR_YAW})")
    parser.add_argument("--smooth_scale", type=float, default=SMOOTHING_FACTOR_SCALE,
                        help=f"Exponential smoothing factor for scale (0-1, 0 or None to disable, default: {SMOOTHING_FACTOR_SCALE})")

    args = parser.parse_args()

    MIN_CONFIDENCE = args.confidence
    SMOOTHING_FACTOR_YAW = args.smooth_yaw if args.smooth_yaw and 0 < args.smooth_yaw < 1 else None
    SMOOTHING_FACTOR_SCALE = args.smooth_scale if args.smooth_scale and 0 < args.smooth_scale < 1 else None

    process_pose_data(args.input_json, args.output_json, args.gender)