import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import os
import pandas as pd

from tracker import SportsTracker
from transformation import PerspectiveTransformer
from analytics import AnalyticsEngine

# Page Config
st.set_page_config(page_title="AI Sports Analytics", layout="wide")

st.title(" AI-Powered Sports Analytics")

# --- Sidebar ---
st.sidebar.header("Settings")

# File Uploader
video_file = st.sidebar.file_uploader("Upload Match Video", type=['mp4', 'avi', 'mov'])

# Model Settings
conf_thres = st.sidebar.slider("YOLO Confidence Threshold", 0.1, 1.0, 0.3)
max_age = st.sidebar.slider("SORT Max Age (Frames)", 1, 60, 30)

# --- Logic ---

if video_file is not None:
    # Save uploaded file to temp
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    video_path = tfile.name

    # Initialize Modules
    with st.spinner("Initializing AI Models..."):
        try:
            tracker = SportsTracker(conf_thres=conf_thres, max_age=max_age)
            transformer = PerspectiveTransformer()
            analytics = AnalyticsEngine()
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            st.stop()

    # Layout
    col_video, col_map = st.columns([2, 1])
    
    with col_video:
        st.subheader("Real-time Tracking")
        video_placeholder = st.empty()
        
    with col_map:
        st.subheader("Tactical Map")
        map_placeholder = st.empty()

    # Metrics
    st.markdown("### Real-time Metrics")
    m1, m2, m3 = st.columns(3)
    metric_fps = m1.empty()
    metric_count = m2.empty()
    metric_status = m3.empty()

    # Process Video
    if st.button("Start Analysis"):
        cap = cv2.VideoCapture(video_path)
        
        all_tracks_log = []
        frame_idx = 0
        
        # Minimap static background
        minimap_bg = np.zeros((340, 525, 3), dtype=np.uint8) # 5x scale of 105x68 approx
        minimap_bg[:] = (0, 100, 0) # Green
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            start_time = time.time()
            
            # 1. Track
            annotated_frame, tracks = tracker.process_frame(frame)
            
            # 2. Transform & Map
            current_frame_points = []
            
            # Create a fresh map for this frame
            curr_map = minimap_bg.copy()
            
            for t in tracks:
                # t: [x1, y1, x2, y2, id]
                bbox = t[:4]
                tid = int(t[4])
                
                # Transform
                pitch_coord = transformer.transform_bbox_base(bbox) # returns (x, y)
                
                # Log
                all_tracks_log.append([frame_idx, tid, pitch_coord[0], pitch_coord[1]])
                current_frame_points.append(pitch_coord)
                
                # Draw on Map (Scale 105x68 -> 525x340)
                mx = int(pitch_coord[0] * 5)
                my = int(pitch_coord[1] * 5)
                
                # Clip to map
                mx = max(0, min(524, mx))
                my = max(0, min(339, my))
                
                cv2.circle(curr_map, (mx, my), 5, (0, 0, 255), -1) # Red Player
                
            # CalcFPS
            fps = 1.0 / (time.time() - start_time + 0.001)
            
            # Display
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            map_rgb = cv2.cvtColor(curr_map, cv2.COLOR_BGR2RGB)
            
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            map_placeholder.image(map_rgb, channels="RGB", caption="Top-Down View", use_container_width=True)
            
            metric_fps.metric("FPS", f"{fps:.1f}")
            metric_count.metric("Active Players", len(tracks))
            metric_status.info(f"Processing Frame {frame_idx}")
            
            frame_idx += 1
            
            # Limit frame rate for UI responsiveness (optional)
            # time.sleep(0.01)

        cap.release()
        metric_status.success("Processing Complete!")
        
        # --- Post Analysis ---
        st.markdown("---")
        st.header("Match Analytics")
        
        if all_tracks_log:
            # Prepare data
            df = pd.DataFrame(all_tracks_log, columns=["Frame", "ID", "X", "Y"])
            
            # 1. Download CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Tracking Data (CSV)",
                csv,
                "match_tracking.csv",
                "text/csv",
                key='download-csv'
            )
            
            col_heat, col_hull = st.columns(2)
            
            # 2. Heatmap
            with col_heat:
                st.subheader("Player Heatmap")
                # Use all points
                points = df[['X', 'Y']].values.tolist()
                fig_heat = analytics.generate_heatmap(points)
                st.pyplot(fig_heat)

            # 3. Convex Hull
            with col_hull:
                st.subheader("Team Shape (End of Match)")
                # Use points from last processed frame
                last_frame = df[df['Frame'] == frame_idx - 1]
                last_points = last_frame[['X', 'Y']].values.tolist()
                fig_hull = analytics.plot_convex_hull(last_points)
                st.pyplot(fig_hull)
        else:
            st.warning("No tracking data generated.")

else:
    st.info("Please upload a video file to begin.")
