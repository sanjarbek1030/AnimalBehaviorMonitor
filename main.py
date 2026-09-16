"""
==================================================================
 ANIMAL BEHAVIOR MONITOR
==================================================================
What this script does (in plain English):
  1. Opens a video file called 'input_video.mp4'.
  2. Runs each frame through a pre-trained YOLOv8 object detector
     (with built-in tracking) to find animals like dogs, cats,
     cows, horses, etc.
  3. Follows each detected animal over time (using a "track ID")
     and looks at how much its bounding box has moved recently to
     guess a simple behavior:
         - "Sleeping/Resting"  -> barely moving for a while
         - "Running/Abnormal"  -> moving very fast between frames
         - "Eating"            -> standing inside a defined "feeding
                                   zone" rectangle in the frame
         - "Standing/Idle"     -> anything else (default/normal)
  4. Draws a box + label ("Dog - Running") on every frame.
  5. Saves everything to 'output_video.mp4'.

This is a RULE-BASED system, not a trained behavior classifier.
It's a great starting point / demo. For real accuracy you would
eventually train a custom model on labeled behavior data.
==================================================================
"""

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# ------------------------------------------------------------------
# SECTION 1: CONFIGURATION (tweak these values to fit your video)
# ------------------------------------------------------------------

INPUT_VIDEO_PATH = "input_video.mp4"
OUTPUT_VIDEO_PATH = "output_video.mp4"

# Pre-trained YOLOv8 model. "yolov8n.pt" = "nano" = smallest/fastest.
# The first time you run this, Ultralytics will auto-download the
# weights file for you (needs an internet connection once).
MODEL_PATH = "yolov8n.pt"

# Only detect these COCO classes (YOLOv8n is trained on the COCO
# dataset, which already includes common animals). We filter to
# animals only so we ignore people, cars, etc.
# COCO class id -> name (the ones we care about):
ANIMAL_CLASS_IDS = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}

# Minimum confidence score (0.0 - 1.0) for a detection to count.
CONFIDENCE_THRESHOLD = 0.4

# How many recent frames of movement history to keep per animal.
# A bigger number = smoother/more stable behavior decisions, but
# slower to react to changes.
HISTORY_LENGTH = 30

# --- Movement thresholds (in pixels) ---
# These control how "twitchy" vs "chill" the behavior detector is.
# You will likely need to adjust these based on your video's
# resolution and how close/far the animals are from the camera.

# If the animal's box center has barely moved (average distance
# between consecutive frames) over the recent history, call it
# "Sleeping/Resting".
RESTING_MOVEMENT_THRESHOLD = 3.0

# If the box center jumps more than this many pixels between two
# consecutive frames, call it "Running/Abnormal Movement".
RUNNING_MOVEMENT_THRESHOLD = 40.0

# --- "Eating zone" ---
# A simple rectangle in the frame that represents, e.g., a food
# trough or feeding area. If an animal's box center falls inside
# this rectangle, we label it as "Eating".
# Format: (x1, y1, x2, y2) in pixel coordinates.
# NOTE: You MUST adjust these coordinates to match YOUR video!
# A quick way to find good coordinates: open one frame of your
# video in any image viewer and note where the feeding area is.
EATING_ZONE = (100, 300, 400, 480)

# Should we draw the eating zone rectangle on the output video?
# Useful for debugging/tuning EATING_ZONE.
DRAW_EATING_ZONE = True


# ------------------------------------------------------------------
# SECTION 2: HELPER FUNCTIONS
# ------------------------------------------------------------------

def get_box_center(x1, y1, x2, y2):
    """Given a bounding box's corners, return its center point (cx, cy)."""
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy


def point_in_rect(px, py, rect):
    """Check whether point (px, py) lies inside rectangle (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2


def classify_behavior(center_history, current_center, eating_zone):
    """
    Decide a simple behavior label based on:
      - center_history: a deque of past (cx, cy) points for this animal
      - current_center: the (cx, cy) of the animal in THIS frame
      - eating_zone: the (x1, y1, x2, y2) feeding rectangle

    Returns a string label such as "Sleeping/Resting", "Running/Abnormal
    Movement", "Eating", or "Standing/Idle".
    """

    # --- Rule 1: Eating (highest priority visual cue - a fixed zone) ---
    if point_in_rect(current_center[0], current_center[1], eating_zone):
        return "Eating"

    # If we don't have at least 2 points yet, we can't measure movement,
    # so just call it idle for now.
    if len(center_history) < 2:
        return "Standing/Idle"

    # --- Rule 2: Running / Abnormal movement ---
    # Look at the jump between the very last two recorded positions.
    last_point = center_history[-1]
    prev_point = center_history[-2]
    frame_to_frame_distance = np.hypot(
        last_point[0] - prev_point[0],
        last_point[1] - prev_point[1],
    )
    if frame_to_frame_distance > RUNNING_MOVEMENT_THRESHOLD:
        return "Running/Abnormal Movement"

    # --- Rule 3: Sleeping / Resting ---
    # Only make this decision once we have a good chunk of history,
    # so a single still frame doesn't instantly get called "sleeping".
    if len(center_history) >= HISTORY_LENGTH:
        distances = []
        points = list(center_history)
        for i in range(1, len(points)):
            d = np.hypot(
                points[i][0] - points[i - 1][0],
                points[i][1] - points[i - 1][1],
            )
            distances.append(d)
        average_movement = sum(distances) / len(distances)

        if average_movement < RESTING_MOVEMENT_THRESHOLD:
            return "Sleeping/Resting"

    # --- Default fallback ---
    return "Standing/Idle"


# ------------------------------------------------------------------
# SECTION 3: LOAD THE MODEL
# ------------------------------------------------------------------

print("Loading YOLOv8 model... (this may download weights on first run)")
model = YOLO(MODEL_PATH)


# ------------------------------------------------------------------
# SECTION 4: OPEN THE INPUT VIDEO
# ------------------------------------------------------------------

cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(
        f"Could not open '{INPUT_VIDEO_PATH}'. "
        f"Make sure the file exists in your PyCharm project folder."
    )

# Grab video properties so our output video matches the input.
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0  # sensible fallback if the video file doesn't report FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Input video: {frame_width}x{frame_height} @ {fps:.2f} FPS")


# ------------------------------------------------------------------
# SECTION 5: SET UP THE OUTPUT VIDEO WRITER
# ------------------------------------------------------------------

# 'mp4v' is a widely-compatible codec for writing .mp4 files.
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    OUTPUT_VIDEO_PATH,
    fourcc,
    fps,
    (frame_width, frame_height),
)


# ------------------------------------------------------------------
# SECTION 6: TRACK MOVEMENT HISTORY PER ANIMAL
# ------------------------------------------------------------------

# This dictionary stores, for each animal's track ID, a short memory
# (deque) of its recent center-point positions. deque with maxlen
# automatically drops old points once it's full - very convenient!
track_histories = {}


# ------------------------------------------------------------------
# SECTION 7: MAIN PROCESSING LOOP
# ------------------------------------------------------------------

frame_number = 0
print("Processing video... this may take a while depending on video length.")

while True:
    success, frame = cap.read()
    if not success:
        # No more frames left -> we've reached the end of the video.
        break

    frame_number += 1

    # --- Run YOLOv8 tracking on this frame ---
    # model.track() runs detection AND assigns a persistent ID to each
    # object across frames, which is exactly what we need to measure
    # movement over time. persist=True tells it to remember tracks
    # between calls (i.e., across frames of this same video).
    results = model.track(
        frame,
        persist=True,
        conf=CONFIDENCE_THRESHOLD,
        classes=list(ANIMAL_CLASS_IDS.keys()),
        verbose=False,
    )

    result = results[0]

    # If YOLO found any boxes in this frame, 'result.boxes' will hold them.
    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()       # bounding box corners
        track_ids = result.boxes.id.cpu().numpy().astype(int)  # tracker IDs
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # class ids
        confidences = result.boxes.conf.cpu().numpy()  # confidence scores

        for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confidences):
            x1, y1, x2, y2 = box
            center = get_box_center(x1, y1, x2, y2)

            # Get (or create) this animal's movement history.
            if track_id not in track_histories:
                track_histories[track_id] = deque(maxlen=HISTORY_LENGTH)

            history = track_histories[track_id]

            # Decide the behavior BEFORE adding the new point, so
            # "frame-to-frame jump" comparisons make sense.
            behavior = classify_behavior(history, center, EATING_ZONE)

            # Now record this frame's position for future frames.
            history.append(center)

            # Look up a human-readable animal name.
            animal_name = ANIMAL_CLASS_IDS.get(class_id, "Animal")

            # --- Draw the bounding box ---
            top_left = (int(x1), int(y1))
            bottom_right = (int(x2), int(y2))
            box_color = (0, 255, 0)  # green (BGR format)
            cv2.rectangle(frame, top_left, bottom_right, box_color, 2)

            # --- Draw the label text (e.g., "Dog - Running/Abnormal Movement") ---
            label = f"{animal_name} (ID {track_id}) - {behavior}"
            text_position = (int(x1), max(int(y1) - 10, 20))
            cv2.putText(
                frame,
                label,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    # --- Optionally draw the "eating zone" rectangle for reference ---
    if DRAW_EATING_ZONE:
        ex1, ey1, ex2, ey2 = EATING_ZONE
        cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (255, 0, 0), 2)
        cv2.putText(
            frame,
            "Feeding Zone",
            (ex1, max(ey1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # --- Write this processed frame to the output video ---
    out.write(frame)

    if frame_number % 50 == 0:
        print(f"  ...processed {frame_number} frames")


# ------------------------------------------------------------------
# SECTION 8: CLEAN UP (very important - prevents corrupted video files!)
# ------------------------------------------------------------------

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Done! Processed {frame_number} frames.")
print(f"Output saved to: {OUTPUT_VIDEO_PATH}")
