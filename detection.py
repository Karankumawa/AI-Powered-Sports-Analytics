from ultralytics import YOLO
import cv2
import numpy as np

class SportsDetector:
    def __init__(self, model_path='yolov8n.pt', conf_thres=0.3):
        """
        Initialize the SportsDetector with a YOLOv8 model.
        
        Args:
            model_path: Path to the .pt model file.
            conf_thres: Confidence threshold for filtering detections.
        """
        print(f"Loading YOLOv8 model from {model_path}...")
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres

    def detect(self, frame):
        """
        Run detection on a single frame.
        
        Args:
            frame: Numpy array (image).
            
        Returns:
            detections: Numpy array of shape (N, 6) -> [x1, y1, x2, y2, score, class_id]
        """
        results = self.model(frame, verbose=False, conf=self.conf_thres)
        
        # Extract boxes
        # results[0].boxes.data returns [x1, y1, x2, y2, conf, cls]
        if len(results) > 0 and results[0].boxes is not None:
            # Move to CPU and numpy
            dets = results[0].boxes.data.cpu().numpy() 
            return dets
        else:
            return np.empty((0, 6))

if __name__ == "__main__":
    # Test stub
    # Create a blank image
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    detector = SportsDetector()
    try:
        dets = detector.detect(img)
        print("Detection ran successfully.")
        print("Detections shape:", dets.shape)
    except Exception as e:
        print(f"Detection failed: {e}")
        print("Make sure 'ultralytics' is installed.")
