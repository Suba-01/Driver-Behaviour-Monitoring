# README.md

# Driver & Pedestrian Monitoring (Combine\_code.ipynb)

Real-time driver-monitoring and pedestrian detection notebook.
This project combines driver drowsiness detection (Eye Aspect Ratio), emotion-based alerting (using DeepFace), and pedestrian detection (YOLOv5 via torch.hub) in a single Jupyter notebook. It listens to two camera feeds (driver-facing and external/road) and raises an audible alarm when drowsiness, aggressive emotion, or pedestrians are detected.

## Features

* Camera scan helper to list available camera indices (0..9)
* Driver monitoring using dlib facial landmarks and Eye Aspect Ratio (EAR)
* Configurable thresholds:

  * `EYE_AR_THRESH = 0.3`
  * `EYE_AR_CONSEC_FRAMES = 48`
* Emotion detection of the driver using DeepFace. If emotion is "angry", triggers an alarm and shows "Aggressive Driving"
* Pedestrian detection using YOLOv5 (`ultralytics/yolov5` via `torch.hub`)
* Audible alarm (Windows `winsound.Beep()` used in the notebook)

## Files

* `Combine_code.ipynb` — main Jupyter notebook
* `requirements.txt` — recommended dependencies
* `LICENSE` — MIT license

## Requirements

* Python 3.8+
* Webcam(s) or USB cameras
* CPU or GPU with PyTorch support (GPU recommended for faster YOLO inference)

Python packages (see `requirements.txt`):

```
opencv-python
dlib
deepface
torch
torchvision
scipy
imutils
pandas
numpy
```

**Notes:**

* Follow the official PyTorch installation instructions for your OS/CUDA configuration.
* `dlib` can be tricky to install; if `pip install` fails, try `conda install -c conda-forge dlib`.
* DeepFace may pull heavy ML dependencies (TensorFlow or PyTorch backends).

## Configuration

Edit variables in the notebook as needed:

* **Shape predictor path**

```python
shape_predictor_path = "path/to/shape_predictor_68_face_landmarks.dat"
```

* **Camera indices**

```python
cap_driver = cv2.VideoCapture(1)        # driver-facing camera
cap_pedestrian = cv2.VideoCapture(0)    # road/external camera
```

* **EAR & timing constants**

```python
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48
```

* **YOLO model**

```python
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True, trust_repo=True)
```

## How it works

1. Initializes two camera feeds.
2. **Driver camera:**

   * Detect face using dlib.
   * Compute EAR; if below threshold for consecutive frames → alarm.
   * Analyze emotion using DeepFace; if "angry" → alarm.
3. **Pedestrian camera:**

   * Run YOLOv5 to detect people; triggers alarm if detected.
4. Press `q` to quit; releases cameras and closes OpenCV windows.

## How to run

### 1. Using Jupyter Notebook

```bash
# Optional: create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook Combine_code.ipynb
```

### 2. Convert to Python script

```bash
jupyter nbconvert --to script Combine_code.ipynb
python Combine_code.py
```

## Cross-platform notes

* **Audio:** Replace `winsound.Beep()` with `playsound` or `simpleaudio` on non-Windows systems.
* **dlib:** If installation fails, try `conda install -c conda-forge dlib`.
* **YOLOv5:** Requires internet on first run.
* **DeepFace:** Downloads backend models on first run.
* **Performance:** CPU inference may be slow; GPU recommended.
* **Permissions:** Ensure Python has access to webcams.

## Known Issues & Suggestions

* Hard-coded path to shape predictor
* Windows-only audible alerts
* Heavy ML dependencies (dlib, DeepFace)
* YOLOv5 CPU inference can be slow
* Consider frame throttling and resizing for performance

## Improvements / TODO

* Add CLI options & convert notebook to modular Python script
* Add configuration file (YAML/JSON)
* Replace winsound with cross-platform audio library
* Unit tests for EAR computation
* Add recording/logging feature

## Privacy & Legal

* Captures video & analyzes faces; ensure consent and comply with privacy laws.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

* Subashini Manickam
