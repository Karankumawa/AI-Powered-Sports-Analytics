import os
import csv
from typing import List, Tuple

def parse_sportsmot_annotation(file_path: str) -> List[dict]:
    """
    Parses a SportsMOT annotation file.
    Format is expected to be: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, ...
    
    Args:
        file_path: Path to the annotation text file.
        
    Returns:
        List of dictionaries containing parsed annotation data.
    """
    annotations = []
    if not os.path.exists(file_path):
        print(f"Warning: Annotation file not found at {file_path}")
        return annotations

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # Skip empty lines or malformed lines
            if not row or len(row) < 6:
                continue
            
            try:
                # SportsMOT format: frame_idx, track_id, left, top, width, height, conf, class, visibility
                # We only strictly need the first 6 for basic training
                frame_idx = int(row[0])
                track_id = int(row[1])
                bb_left = float(row[2])
                bb_top = float(row[3])
                bb_width = float(row[4])
                bb_height = float(row[5])
                
                annotations.append({
                    "frame": frame_idx,
                    "id": track_id,
                    "x": bb_left,
                    "y": bb_top,
                    "w": bb_width,
                    "h": bb_height
                })
            except ValueError:
                continue
                
    return annotations

def convert_to_yolo_format(annotations: List[dict], output_dir: str, img_width: int = 1920, img_height: int = 1080):
    """
    Converts parsed annotations to YOLOv8 format and saves them as .txt files per frame.
    YOLO format: <class_id> <x_center> <y_center> <width> <height> (all normalized 0-1)
    
    Args:
        annotations: List of annotation dicts.
        output_dir: Directory to save the label files.
        img_width: Width of the image (for normalization).
        img_height: Height of the image (for normalization).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Group by frame
    frames = {}
    for ann in annotations:
        f_idx = ann['frame']
        if f_idx not in frames:
            frames[f_idx] = []
        frames[f_idx].append(ann)
        
    # Write to files
    class_id = 0 # Assuming single class 'player' or 'ball' mapped to 0 for now. 
                 # SportsMOT usually separates classes, but for this project we'll assume a standard class 0.
    
    for f_idx, anns in frames.items():
        # GT file naming convention usually text file per image name
        # Here we'll name it frame_{f_idx:06d}.txt
        filename = f"frame_{f_idx:06d}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            for ann in anns:
                # Calculate center and normalize
                x_center = (ann['x'] + ann['w'] / 2) / img_width
                y_center = (ann['y'] + ann['h'] / 2) / img_height
                n_width = ann['w'] / img_width
                n_height = ann['h'] / img_height
                
                # Clip to [0, 1] just in case
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                n_width = max(0.0, min(1.0, n_width))
                n_height = max(0.0, min(1.0, n_height))
                
                line = f"{class_id} {x_center:.6f} {y_center:.6f} {n_width:.6f} {n_height:.6f}\n"
                f.write(line)

if __name__ == "__main__":
    # Example usage
    # Create a dummy annotation file for testing
    dummy_input = "test_gt.txt"
    with open(dummy_input, "w") as f:
        f.write("1,1,100,200,50,100,1,1,1\n")
        f.write("1,2,300,400,60,120,1,1,1\n")
        f.write("2,1,105,205,50,100,1,1,1\n")
    
    print("Parsing...")
    data = parse_sportsmot_annotation(dummy_input)
    print(f"Parsed {len(data)} objects.")
    
    print("Converting to YOLO...")
    convert_to_yolo_format(data, "output_labels", 1920, 1080)
    print("Done. Check 'output_labels' directory.")
    
    # Cleanup
    if os.path.exists(dummy_input):
        os.remove(dummy_input)
