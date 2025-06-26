from gz.transport13 import Node
import gz.msgs10.image_pb2 as image_pb2
import cv2
import numpy as np
import apriltag
from apriltag import DetectorOptions
import os

class CameraAprilTagDetector:
    def __init__(self, camera_topic, simulation_state_checker,
                 camera_params=None,   # (fx, fy, cx, cy)
                 tag_size_m=None,
                 instance_id=None):
        """
        Initializes the camera subscriber and AprilTag detector.
        """
        self.instance_id = instance_id
        self.camera_topic = camera_topic
        self._is_simulation_running = simulation_state_checker
        partition_name = f"sim{instance_id}"
        os.environ["GZ_PARTITION"] = partition_name
        self.camera_node = Node()

        # Initialize detector options
        detector_options = DetectorOptions(
            families='tag36h11 tag25h9',
            border=1,
            nthreads=1,
            quad_decimate=1.0,
            quad_blur=0.0,
            refine_edges=True,
            refine_decode=False,
            refine_pose=False,
            debug=False,
            quad_contours=True
        )
        self._cam_params = camera_params
        self.tag_size = tag_size_m  # Save tag size if provided
        self.detector = apriltag.Detector(options=detector_options)

        print(f"[*] Subscribing to camera topic: {self.camera_topic}")
        # Use the provided camera_topic instead of the hard-coded one
        self.camera_node.subscribe(topic=self.camera_topic,
                                   msg_type=image_pb2.Image,
                                   callback=self.camera_callback)

        # Initialize tag detection results
        self.tag_found = 0
        self.tag_dist_x = 300
        self.tag_dist_y = 300
        self.camera_data = np.array([self.tag_found, self.tag_dist_x, self.tag_dist_y], dtype=np.float32)

    def camera_callback(self, msg):
        # Check if simulation is running before processing
        if not self._is_simulation_running():
            return

        img_bytes = msg.data
        width = msg.width
        height = msg.height

        try:
            # Determine if the image is provided in color (3 channels) or as grayscale
            image_np = np.frombuffer(img_bytes, dtype=np.uint8)
            if image_np.size == height * width * 3:
                image_np = image_np.reshape((height, width, 3))
                # Convert color to grayscale for detection
                image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                image_gray = image_np.reshape((height, width))

            # Detect AprilTags on the grayscale image
            results = self.detector.detect(image_gray)

            if results:
                # Use the first detected tag for offset calculations
                tag = results[0]
                # Compute offset from image center
                cx_img, cy_img = width / 2.0, height / 2.0
                self.tag_dist_x = tag.center[0] - cx_img
                self.tag_dist_y = tag.center[1] - cy_img
                self.tag_found = 1
                #print(f"[*] Tag detected at offset: x={self.tag_dist_x}, y={self.tag_dist_y}")
            else:
                self.tag_found = 0

            self.camera_data = np.array([self.tag_found, self.tag_dist_x, self.tag_dist_y], dtype=np.float32)

            # Optionally, display the image with detections (if image is in color)
            """           
            if image_gray.ndim == 2 and image_np.ndim == 3:
                display_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                if results:
                    for r in results:
                        corners = r.corners.astype(int)
                        cv2.polylines(display_image, [corners], True, (0, 255, 0), 2)
                        center = tuple(r.center.astype(int))
                        cv2.circle(display_image, center, 5, (0, 0, 255), -1)
                        cv2.putText(display_image, str(r.tag_id), (center[0]-10, center[1]-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow(f"Camera Feed with Detections SIM: {self.instance_id}", display_image)
                cv2.waitKey(1)
                """ 

        except Exception as e:
            print(f"[!] Error processing camera data from {self.camera_topic}: {e}")