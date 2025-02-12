<!-- omit from toc -->
# Traffic Counting App

A Flask-based web app that processes uploaded video files to perform object detection, counting, and optional annotated video export. It uses YOLO (Ultralytics) for detection and can create Excel reports with the results.

- [Overview](#overview)
  - [`script.py`](#scriptpy)
  - [`app.py`](#apppy)
  - [`utils`](#utils)
    - [`data.py`](#datapy)
    - [`session.py`](#sessionpy)
    - [`tracking.py`](#trackingpy)
    - [`export/`](#export)
- [Installation](#installation)
- [Usage](#usage)
- [Notes](#notes)

## Overview

### [`script.py`](script.py)

[`script.py`](script.py) provides the core processing logic without any GUI code (except a cv2 window for tripline drawing).  
It handles opening a video, detecting objects, tracking them, counting crossings, generating Excel reports, and optionally exporting annotated videos.  

### [`app.py`](app.py)

[`app.py`](app.py) uses Flask to provide a browser-based interface and asynchronous server handling. This allows the server to be deployed separately from the client machines accessing it.

### [`utils`](utils)

Both implementations rely on the same framework and processing tools, found in [`utils`](utils).

The utils directory contains several key modules:

#### `data.py`
  
  Core data management class that handles:
  
- Video metadata and parameters
- Model configuration and selection
- Tracking data storage
- Site and timing information
- Export settings

#### `session.py`
  
  Manages user sessions with features for:

- Session creation and deletion
- Data persistence
- Multiple concurrent user support
- Progress tracking

#### `tracking.py`
  
  Implements object detection and tracking with:

- YOLO model integration
- Object trajectory analysis
- Tripline crossing detection
- Classification confidence scoring

#### `export/`
  
  Contains export-related modules:

- **`video.py`**: Handles annotated video creation with:
  - Trajectory visualization
  - Bounding box drawing
  - Real-time statistics display
  - FFmpeg integration for video encoding

- **`xlsx.py`**: Manages Excel report generation with:
  - Detailed crossing data (Direct counting output report)
  - Vehicle classification summaries (Compiled report)
  - Multiple report compilation
  - Compiling of Street Count app output to report format

---

- This was made with the expectation of Ultralytics' YOLO models and relies on the associated libraries and tools first and foremost.
- Both implementations create a local copy of all uploads (video & model), as well as log input parameters, for the sake of trouble shooting and to ensure data integrity (in case processing is interrupted).

---

## Installation

1. Clone the repository.  

    ```bash
    git clone https://github.com/AmauryDufour/TrafficCounting.git 
    ```

2. Create and activate a Python virtual environment.

    ```bash
    python -m venv .venv
    .venv\Scripts\activate # Windows
    source .venv/bin/activate # Unix
    ```

3. Install dependencies:  

    ```bash
    pip install -r requirements.txt
    ```

4. By default, the FFmpeg executable used is the one referenced by your local `ffmpeg` PATH variable (refer to the ffmpeg doc [here](https://www.ffmpeg.org/download.html)). Add a .env file in the root directory with the following content to specify the path to the FFmpeg executable:

    ```bash
    ffmpeg="path/to/ffmpeg"
    ```

    - For example, on Windows, this is typically `C:\ffmpeg\bin\ffmpeg.exe`.

---

## Usage

Development/testing only. Use a production server for actual deployment *(refer to [Flask - Deploying to Production](https://flask.palletsprojects.com/en/stable/deploying/))*

1. Run the Flask app:

    ```bash
    flask run
    ```

    - This starts a local development server on the default host/port (see *[Flask - Quickstart](https://flask.palletsprojects.com/en/stable/quickstart/#debug-mode)*).
1. Access the web interface (e.g., [127.0.0.1:5000](http://127.0.0.1:5000)) to upload video and model files ( `.pt`, `openvino` and `onnx` supported), draw triplines, set direction names, and process the video.
1. The web interface also features other utilities apart from the main processing function :
   1. **Compiler** handles the processing of one/multiple traffic counting reports (which identify each vehicle tripline crossing individually) to output a condensed version, with totals *per site per direction per 15-min interval per vehicle class*
   1. **History** allows the user to go through all logged records of past sessions (whether processing succesfully concluded or not). The session id displayed at the bottom of the page for each processing session is useful to this aim.
   1. **Street Count** allows the user to transform the `.csv` output of the [Street Count app by Neil Kimmet](https://streetcount.app/) to the same compiled report format as this app.

- The backend logic and processing is handled in separate threads by [`app.py`](app.py) : multiple current processes can be handled at once (performance is however degraded)

---

## Notes

- The Flask server can be run on a separate machine so multiple users can access it for video processing tasks.
- All the graphical interactions (tripline drawing, form inputs, etc.) are handled by the web app.
- **[`script.py`](script.py)** can be used in two ways :

  - Imported and used as such :

    ```python
        >>>from script import *
        >>>params = {}
        >>>params['video_path'] = r"\path\to\your\vid"
        >>>params['model_path'] = r"\path\to\your\model"
        >>>params['site_location'] = "Name of Location"
        >>>params['inference_tracker'] = "tracker.yaml" # 2 are supported : `bytetrack.yaml` & `botsort.yaml` (BoT-SORT is slower)
        >>>params['export_video'] = True # False 
        >>>params['start_date'] = "2025-01-20" # 'YYYY-MM-DD'
        >>>params['start_time'] = "12:12" # 'HH:MM'
        >>>params['ffmpeg_executable_path'] = "ffmpeg"
        >>>run(params)
    ```

- The contents of `if __name__ == "__main__:"` can be edited and the script directly run : `python script.py`
