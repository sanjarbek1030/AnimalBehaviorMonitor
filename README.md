# 🐾 Animal Behavior Monitor

A beginner-friendly computer vision tool that detects animals in video footage and infers simple behaviors — **Sleeping/Resting**, **Eating**, **Running/Abnormal Movement**, or **Standing/Idle** — using [YOLOv8](https://github.com/ultralytics/ultralytics) object detection/tracking and rule-based movement analysis.

No custom training required. Point it at a video, get back an annotated video with bounding boxes and behavior labels.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-purple" alt="YOLOv8">
  <img src="https://img.shields.io/badge/OpenCV-Video%20I%2FO-green" alt="OpenCV">
</p>

---

## ✨ Features

- **Detects common animals** — dog, cat, cow, horse, sheep, bird, elephant, bear, zebra, giraffe (via COCO classes)
- **Persistent tracking** — follows each animal across frames using YOLOv8's built-in tracker, so behavior is based on *that specific animal's* history, not one-off frames
- **Rule-based behavior inference**:
  | Behavior | Trigger |
  |---|---|
  | 🍽️ Eating | Animal's center point is inside a defined "feeding zone" rectangle |
  | 🏃 Running/Abnormal Movement | Bounding box center shifts drastically between consecutive frames |
  | 😴 Sleeping/Resting | Bounding box barely moves over many recent frames |
  | 🧍 Standing/Idle | Default fallback when none of the above apply |
- **Visual overlay** — bounding boxes + labels like `Dog (ID 3) - Running/Abnormal Movement` drawn on every frame
- **Simple config block** — all thresholds and paths are constants at the top of the script, no need to dig through the code to tune it

## ⚠️ Limitations

This is a **rule-based heuristic system**, not a trained behavior classifier. It infers behavior purely from bounding-box movement and a fixed feeding-zone rectangle — it doesn't actually understand posture, chewing motion, or context. Expect to tune thresholds per video, and treat outputs as a helpful starting point rather than ground truth. For production-grade accuracy, you'd eventually want a custom-trained action-recognition model.

## 📦 Requirements

- Python 3.8+
- [Ultralytics](https://pypi.org/project/ultralytics/) (YOLOv8)
- [OpenCV](https://pypi.org/project/opencv-python/)
- NumPy (installed automatically as a dependency)

## 🚀 Installation

```bash
git clone https://github.com/your-username/animal-behavior-monitor.git
cd animal-behavior-monitor
pip install ultralytics opencv-python
```

> The first run will automatically download the `yolov8n.pt` model weights (~6 MB) — an internet connection is required once.

## ▶️ Usage

1. Place your source footage in the project root as **`input_video.mp4`**.
2. Run the script:

   ```bash
   python animal_behavior_monitor.py
   ```

3. The annotated result is saved as **`output_video.mp4`** in the same folder.

### Running in PyCharm

1. Add `animal_behavior_monitor.py` and `input_video.mp4` to your project root.
2. Open the **Terminal** tab and run `pip install ultralytics opencv-python`.
3. Right-click the script → **Run 'animal_behavior_monitor'**.
4. Check the Run window for progress; open `output_video.mp4` when it finishes.

## ⚙️ Configuration

All tunable settings live in the config block near the top of `animal_behavior_monitor.py`:

```python
INPUT_VIDEO_PATH = "input_video.mp4"
OUTPUT_VIDEO_PATH = "output_video.mp4"
MODEL_PATH = "yolov8n.pt"

CONFIDENCE_THRESHOLD = 0.4       # minimum detection confidence
HISTORY_LENGTH = 30              # frames of movement history kept per animal

RESTING_MOVEMENT_THRESHOLD = 3.0   # px — below this avg movement = Resting
RUNNING_MOVEMENT_THRESHOLD = 40.0  # px — above this frame-to-frame jump = Running

EATING_ZONE = (100, 300, 400, 480)  # (x1, y1, x2, y2) feeding rectangle
DRAW_EATING_ZONE = True             # draw the zone for easy tuning
```

**Tuning tips:**
- Everything labeled "Running"? → raise `RUNNING_MOVEMENT_THRESHOLD`.
- Nothing ever labeled "Sleeping"? → lower `RESTING_MOVEMENT_THRESHOLD` or raise `HISTORY_LENGTH`.
- Adjust `EATING_ZONE` coordinates to match where food/troughs actually appear in your footage — the blue rectangle drawn on the output makes this quick to eyeball and iterate on.

## 🗂️ How It Works

1. **Detect & Track** — Each frame is passed to `model.track()`, which detects animals and assigns a persistent ID to each one across frames.
2. **Record Movement** — Each animal's bounding-box center is stored in a short rolling history (`deque`) keyed by track ID.
3. **Classify Behavior** — A simple rule cascade checks, in order: is it inside the eating zone? did it jump too far since last frame? has it barely moved over recent history? Otherwise, default to idle.
4. **Annotate & Save** — Boxes and labels are drawn on the frame, which is written to `output_video.mp4` via `cv2.VideoWriter`.

## 🛣️ Possible Next Steps

- Train a custom YOLOv8 model on animal-specific classes for better precision
- Replace the rule-based classifier with an actual action-recognition model (e.g., a small temporal CNN/LSTM over pose or optical flow)
- Multi-zone support (multiple feeding/watering/resting areas)
- Export behavior events to a CSV/log for analytics dashboards

## 📄 License

Add your preferred license here (e.g., MIT).# AnimalBehaviorMonitor
