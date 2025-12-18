# AI-Powered Sports Analytics

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

A computer vision project for soccer player tracking and tactical analysis using YOLOv8, SORT, and Homography transformations. Built with a Streamlit interface.

## Features

- **Object Detection**: Uses YOLOv8 to detect players and the ball.
- **Tracking**: Implements SORT (Simple Online and Realtime Tracking) with a Kalman Filter tuned for high-speed sports motion.
- **Tactical Map**: Maps 2D video coordinates to a top-down pitch view using Homography.
- **Analytics**: Generates Player Heatmaps and Team Convex Hulls (shape analysis).
- **Web Interface**: User-friendly dashboard built with Streamlit.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Karankumawa/AI-Powered-Sports-Analytics.git
   cd AI-Powered-Sports-Analytics
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Upload a soccer video file (MP4/AVI) in the sidebar.
3. Adjust tracking parameters (Confidence Threshold, Max Age) if needed.
4. Click **Start Analysis**.

## Project Structure

- `app.py`: Main application entry point.
- `tracker.py`: Core logic for detection and tracking.
- `transformation.py`: Perspective transformation logic.
- `analytics.py`: Visualization and data logging functions.

## Credits

- **SportsMOT Dataset**: Used for inspiration and parameters.
- **Ultralytics**: YOLOv8 model.
