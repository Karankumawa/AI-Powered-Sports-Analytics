import cv2
import numpy as np
import time
from detection import SportsDetector
from tracking import Sort
from transformation import PerspectiveTransformer
from analytics import AnalyticsEngine

def main(video_path='match_video.mp4', output_video='output_video.mp4'):
    # Initialize modules
    print("Initializing modules...")
    
    # 1. Detector
    # Ensure you have a model, e.g., 'yolov8n.pt' or a trained 'best.pt'
    detector = SportsDetector(model_path='yolov8n.pt', conf_thres=0.4)
    
    # 2. Tracker
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    
    # 3. Transformer
    transformer = PerspectiveTransformer()
    
    # 4. Analytics
    analytics = AnalyticsEngine()
    
    # Video Capture
    # Note: If video_path is 0, it uses webcam
    cap = cv2.VideoCapture(video_path if video_path != '0' else 0)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}")
        # Create a dummy blank video stream for demonstration if file missing
        print("Using dummy blank frames for demonstration...")
        use_dummy = True
    else:
        use_dummy = False

    # Video Writer
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if not use_dummy else 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if not use_dummy else 1080
    fps = cap.get(cv2.CAP_PROP_FPS) if not use_dummy else 30
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # Data log
    all_tracks_log = [] # List of (frame_idx, id, x_pitch, y_pitch)
    frame_idx = 0

    print("Starting processing loop. Press 'q' to stop.")
    
    while True:
        if not use_dummy:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # Generate dummy frame
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            # Add some fake "players" (white blobs)
            cv2.circle(frame, (500 + (frame_idx%100)*5, 500), 20, (255,255,255), -1)
            time.sleep(0.03)
            if frame_idx > 100: break

        start_time = time.time()
        
        # --- 1. Detection ---
        detections = detector.detect(frame) 
        # detections: [x1, y1, x2, y2, score, class]
        
        # Prepare for SORT (needs [x1, y1, x2, y2, score])
        if len(detections) > 0:
            dets_to_track = detections[:, :5]
        else:
            dets_to_track = np.empty((0, 5))

        # --- 2. Tracking ---
        # tracks: [x1, y1, x2, y2, track_id]
        tracks = tracker.update(dets_to_track)

        # --- 3. Transformation & 4. Logging ---
        current_frame_pitch_points = []
        for track in tracks:
            # Extract box
            bbox = track[:4]
            track_id = int(track[4])
            
            # Draw bbox and ID on frame
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(track_id), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Perspective Transform
            # Map bottom center of box
            pitch_coord = transformer.transform_bbox_base(bbox)
            
            # Log data
            all_tracks_log.append((frame_idx, track_id, pitch_coord[0], pitch_coord[1]))
            current_frame_pitch_points.append(pitch_coord)

        # Draw Pitch View Overlay (Mini-map)
        # Create a small blank image for minimap
        minimap = np.zeros((200, 300, 3), dtype=np.uint8)
        minimap[:] = (0, 100, 0) # Green field
        # Scale factor: Pitch 105x68 -> Minimap 300x200 (approx scale 2.8)
        scale_x = 300 / 105
        scale_y = 200 / 68
        
        for pc in current_frame_pitch_points:
            mx = int(pc[0] * scale_x)
            my = int(pc[1] * scale_y)
            cv2.circle(minimap, (mx, my), 3, (0, 0, 255), -1) # Red dots for players
        
        # Overlay minimap on main frame (Bottom Left)
        if w > 300 and h > 200:
            frame[h-210:h-10, 10:310] = minimap

        # Save frame
        out.write(frame)
        
        # Display (optional - might not work in headless env, but good for local)
        # cv2.imshow('Sports Analytics', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
            
        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"Processed frame {frame_idx}...")

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete.")
    
    # --- 5. Post-Match Analytics ---
    print("Generating analytics...")
    
    # Extract points for heatmap
    # Using all tracks
    all_points = [(rec[2], rec[3]) for rec in all_tracks_log]
    analytics.generate_heatmap(all_points, output_path="final_heatmap.png")
    
    # Extract points for convex hull (e.g., last frame)
    # Using the last frame's points
    last_frame_points = [(rec[2], rec[3]) for rec in all_tracks_log if rec[0] == frame_idx-1]
    if last_frame_points:
        analytics.plot_convex_hull(last_frame_points, output_path="final_hull.png")
        
    # Log to CSV
    analytics.log_data(all_tracks_log, filename="match_analysis.csv")
    print("Done.")

if __name__ == "__main__":
    main('match_video.mp4')
