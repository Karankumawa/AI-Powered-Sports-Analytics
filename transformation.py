import cv2
import numpy as np

class PerspectiveTransformer:
    def __init__(self):
        """
        Initialize the transformer with hardcoded points.
        In a real application, these should be selected via a GUI or config file.
        """
        # Source points: Pixel coordinates on the image (camera view)
        # Assuming a standard 1920x1080 broadcast frame.
        # These points represent a hypothetical large rectangle on the pitch (e.g., penalty area to center).
        # Top-Left, Top-Right, Bottom-Right, Bottom-Left
        self.src_pts = np.float32([
            [500, 300],   # Top-Left (e.g., far corner of penalty box)
            [1420, 300],  # Top-Right 
            [1600, 900],  # Bottom-Right (near corner)
            [320, 900]    # Bottom-Left
        ])

        # Destination points: Real-world coordinates in meters
        # Standard pitch size: 105m x 68m.
        # Let's map this to a subsection of the pitch or the whole pitch.
        # For this example, let's assume we mapped the whole visible area to the whole pitch 
        # (which is unrealistic but functionally valid for the code structure)
        # or better, map to a 68m width and partial length.
        # Let's assume the 4 points match the 4 corners of the pitch for simplicity of the example.
        # (0,0) is top-left corner of the pitch.
        self.dst_pts = np.float32([
            [0, 0],       # Top-Left
            [105, 0],     # Top-Right (Length: 105m)
            [105, 68],    # Bottom-Right (Width: 68m)
            [0, 68]       # Bottom-Left
        ])

        self.homography_matrix = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        print("Homography Matrix calculated.")

    def transform_point(self, point):
        """
        Transform a point (x, y) from image space to pitch space.
        
        Args:
            point: Tuple or list (x, y)
            
        Returns:
            Tuple (x, y) in pitch coordinates.
        """
        # Convert to homogenous coordinates [x, y, 1]
        p = np.array([[[point[0], point[1]]]], dtype=np.float32)
        
        # Apply perspective transform
        dst = cv2.perspectiveTransform(p, self.homography_matrix)
        
        return (dst[0][0][0], dst[0][0][1])

    def transform_bbox_base(self, bbox):
        """
        Transforms the bottom center of a bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2]
            
        Returns:
            Tuple (x, y) on pitch.
        """
        x_center = (bbox[0] + bbox[2]) / 2
        y_bottom = bbox[3]
        return self.transform_point((x_center, y_bottom))

if __name__ == "__main__":
    pt = PerspectiveTransformer()
    test_pt = (960, 1080) # Bottom center of image
    result = pt.transform_point(test_pt)
    print(f"Image point {test_pt} maps to Pitch coordinates {result}")
