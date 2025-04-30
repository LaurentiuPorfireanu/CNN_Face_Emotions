import cv2
import numpy as np
import json
import time
import os
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime


class PostureDetector:
    def __init__(self, model_path=None, output_dir="posture_data"):
        """
        Initialize the posture detector using OpenPose or MediaPipe.

        Args:
            model_path: Path to OpenPose models (if using OpenPose)
            output_dir: Directory to save posture data
        """
        self.output_dir = output_dir
        Path(output_dir).mkdir(exist_ok=True)

        # Initialize variables that will be set up later
        self.use_mediapipe = False
        self.op = None  # Will hold the OpenPose module if available

        # Keypoint mapping - based on OpenPose format
        self.keypoint_names = [
            "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder",
            "LElbow", "LWrist", "MidHip", "RHip", "RKnee", "RAnkle",
            "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar",
            "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
        ]

        # We'll use MediaPipe since it's more accessible than OpenPose
        try:
            import mediapipe as mp
            self.mp = mp
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
            self.use_mediapipe = True
            print("Using MediaPipe for pose detection")
        except ImportError:
            print("MediaPipe not found. Attempting to use OpenPose...")
            self.use_mediapipe = False
            self.setup_openpose(model_path)

    def setup_openpose(self, model_path):
        """Set up OpenPose if MediaPipe is not available"""
        try:
            # Try to set up OpenPose
            if model_path is None:
                model_path = "models/"  # Default path for OpenPose models

            # Check if OpenPose Python API is available
            try:
                import pyopenpose as op
                self.op = op

                # Custom OpenPose Python API setup
                params = {
                    "model_folder": model_path,
                    "number_people_max": 1,  # We're focusing on one person
                    "net_resolution": "-1x368"
                }

                # Starting OpenPose
                self.opWrapper = op.WrapperPython()
                self.opWrapper.configure(params)
                self.opWrapper.start()

                # Create datums
                self.datum = op.Datum()
                print("Using OpenPose for pose detection")
            except ImportError:
                print("WARNING: Neither MediaPipe nor OpenPose Python API found.")
                print("Will use OpenCV DNN with OpenPose models if available.")

                # Try to use OpenPose with OpenCV DNN module
                try:
                    # Load the network
                    protoFile = os.path.join(model_path, "pose/coco/pose_deploy_linevec.prototxt")
                    weightsFile = os.path.join(model_path, "pose/coco/pose_iter_440000.caffemodel")
                    self.net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

                    if self.net is None:
                        raise Exception("Error loading OpenPose network")

                    print("Using OpenCV DNN with OpenPose models")
                except Exception as e:
                    print(f"ERROR: {e}")
                    print("No pose detection method available. The program will run but won't detect poses.")
        except Exception as e:
            print(f"Error setting up OpenPose: {e}")
            print("No pose detection method available. The program will run but won't detect poses.")

    def process_frame(self, frame):
        """
        Process a single frame to detect posture.

        Args:
            frame: Input image/frame from camera

        Returns:
            processed_frame: Frame with pose overlay
            keypoints: Dictionary with keypoint coordinates and confidence
        """
        if self.use_mediapipe:
            return self._process_with_mediapipe(frame)
        else:
            try:
                if hasattr(self, 'op'):
                    return self._process_with_openpose_api(frame)
                elif hasattr(self, 'net'):
                    return self._process_with_opencv_dnn(frame)
                else:
                    # No pose detection method available
                    return frame, {"error": "No pose detection method available"}
            except Exception as e:
                print(f"Error processing frame: {e}")
                return frame, {"error": str(e)}

    def _process_with_mediapipe(self, frame):
        """Process frame with MediaPipe"""
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        # Draw the pose annotations on the frame
        annotated_frame = frame.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

            # Extract keypoint data
            h, w, _ = frame.shape
            keypoints = {
                "x": [],
                "y": [],
                "confidence": []
            }

            # Convert MediaPipe landmarks to format similar to OpenPose
            landmarks = results.pose_landmarks.landmark

            # MediaPipe has a different keypoint structure than OpenPose
            # We'll map it as closely as possible and fill missing points with zeros
            for i in range(33):  # MediaPipe has 33 landmarks
                if i < len(landmarks):
                    keypoints["x"].append(landmarks[i].x * w)
                    keypoints["y"].append(landmarks[i].y * h)
                    keypoints["confidence"].append(landmarks[i].visibility)
                else:
                    keypoints["x"].append(0)
                    keypoints["y"].append(0)
                    keypoints["confidence"].append(0)

            return annotated_frame, keypoints
        else:
            return annotated_frame, {"error": "No pose detected with MediaPipe"}

    def _process_with_openpose_api(self, frame):
        """Process frame with OpenPose Python API"""
        # Process frame with OpenPose
        self.datum.cvInputData = frame
        self.opWrapper.emplaceAndPop(
            self.op.VectorDatum([self.datum]))  # Using self.op to reference the imported module

        # Check if people were detected
        if self.datum.poseKeypoints is not None and self.datum.poseKeypoints.shape[0] > 0:
            # Extract keypoint data for the first person
            keypoints = {
                "x": [],
                "y": [],
                "confidence": []
            }

            for point_idx in range(self.datum.poseKeypoints.shape[1]):
                keypoints["x"].append(self.datum.poseKeypoints[0, point_idx, 0])
                keypoints["y"].append(self.datum.poseKeypoints[0, point_idx, 1])
                keypoints["confidence"].append(self.datum.poseKeypoints[0, point_idx, 2])

            return self.datum.cvOutputData, keypoints
        else:
            return frame, {"error": "No pose detected with OpenPose API"}

    def _process_with_opencv_dnn(self, frame):
        """Process frame with OpenCV DNN and OpenPose models"""
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        # Input image dimensions for the network
        inHeight = 368
        inWidth = int((inHeight / frameHeight) * frameWidth)

        # Prepare the frame
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(inpBlob)

        # Forward pass
        output = self.net.forward()

        # Extract keypoints
        H = output.shape[2]
        W = output.shape[3]

        keypoints = {
            "x": [],
            "y": [],
            "confidence": []
        }

        # COCO Output Format: keypoints for each person
        for i in range(18):  # COCO model outputs 18 points
            probMap = output[0, i, :, :]
            probMap = cv2.resize(probMap, (frameWidth, frameHeight))

            # Find global maxima of the probMap
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(probMap)

            # Add keypoint if confidence is above threshold
            if maxVal > 0.1:
                keypoints["x"].append(maxLoc[0])
                keypoints["y"].append(maxLoc[1])
                keypoints["confidence"].append(maxVal)
            else:
                keypoints["x"].append(0)
                keypoints["y"].append(0)
                keypoints["confidence"].append(0)

            # Draw on frame
            if maxVal > 0.1:
                cv2.circle(frame, maxLoc, 5, (0, 255, 0), -1)

        # Draw connections between keypoints
        # This would require mapping COCO keypoint pairs

        return frame, keypoints

    def analyze_quality(self, keypoints):
        """
        Analyze the quality of the detected keypoints.
        Similar to Step3_Quality.m logic.

        Args:
            keypoints: Dictionary with keypoint coordinates and confidence

        Returns:
            quality_metrics: Dictionary with PUK and CL metrics
        """
        if "error" in keypoints:
            return {"PUK": 1.0, "CL": 0.0}

        # Calculate proportion of unrecognized keypoints (PUK)
        total_points = len(keypoints["x"])
        unrecognized = sum(1 for i in range(total_points)
                           if keypoints["x"][i] == 0 and keypoints["y"][i] == 0 and keypoints["confidence"][i] == 0)

        puk = unrecognized / total_points if total_points > 0 else 1.0

        # Calculate mean confidence level (CL)
        cl = np.mean(keypoints["confidence"]) if keypoints["confidence"] else 0.0

        return {"PUK": puk, "CL": cl}

    def save_keypoints(self, keypoints, quality, filename=None):
        """
        Save keypoints and quality metrics to JSON file.

        Args:
            keypoints: Dictionary with keypoint coordinates and confidence
            quality: Dictionary with quality metrics
            filename: Optional filename, otherwise uses timestamp
        """
        if "error" in keypoints:
            print(f"Not saving data: {keypoints['error']}")
            return

        # Create data structure
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "keypoints": {
                "x": keypoints["x"],
                "y": keypoints["y"],
                "confidence": keypoints["confidence"]
            },
            "quality": quality
        }

        # Save to JSON file
        if filename is None:
            filename = f"posture_{int(time.time())}.json"

        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved data to {file_path}")

    def detect_posture_issues(self, keypoints):
        """
        Detect potential posture issues based on keypoint relationships.
        This is a simplified version - you could expand this with more sophisticated rules.

        Args:
            keypoints: Dictionary with keypoint coordinates and confidence

        Returns:
            issues: List of detected posture issues
        """
        if "error" in keypoints:
            return ["Unable to detect posture"]

        issues = []

        # Check if we have enough keypoints
        if len(keypoints["x"]) < 25:
            return ["Insufficient keypoints for analysis"]

        # Extract key body points - adjust indices based on the model used
        # This assumes OpenPose/COCO keypoint order
        try:
            # Check shoulder alignment (rounded shoulders)
            left_shoulder_idx = 5
            right_shoulder_idx = 2
            neck_idx = 1

            # Only check if shoulders and neck are detected with sufficient confidence
            shoulder_threshold = 0.5
            if (keypoints["confidence"][left_shoulder_idx] > shoulder_threshold and
                    keypoints["confidence"][right_shoulder_idx] > shoulder_threshold and
                    keypoints["confidence"][neck_idx] > shoulder_threshold):

                # Calculate shoulder line angle relative to horizontal
                shoulder_angle = np.abs(np.degrees(np.arctan2(
                    keypoints["y"][right_shoulder_idx] - keypoints["y"][left_shoulder_idx],
                    keypoints["x"][right_shoulder_idx] - keypoints["x"][left_shoulder_idx]
                )))

                if shoulder_angle > 5:
                    issues.append(f"Uneven shoulders: {shoulder_angle:.1f}° tilt")

                # Check for forward head posture
                # The neck should be relatively straight above the mid-point of shoulders
                mid_shoulder_x = (keypoints["x"][left_shoulder_idx] + keypoints["x"][right_shoulder_idx]) / 2
                forward_head_distance = keypoints["x"][neck_idx] - mid_shoulder_x

                # Positive value indicates head is forward of shoulders
                if forward_head_distance > 30:  # Threshold in pixels, adjust as needed
                    issues.append(f"Forward head posture: {forward_head_distance:.1f}px forward")

            # Check for slouching (based on spine angle)
            # In OpenPose, we can estimate spine using neck (1) and mid-hip (8)
            mid_hip_idx = 8
            if (keypoints["confidence"][neck_idx] > 0.5 and
                    keypoints["confidence"][mid_hip_idx] > 0.5):

                # Calculate spine angle relative to vertical
                spine_vector = [
                    keypoints["x"][neck_idx] - keypoints["x"][mid_hip_idx],
                    keypoints["y"][neck_idx] - keypoints["y"][mid_hip_idx]
                ]
                spine_angle = np.abs(90 - np.degrees(np.arctan2(spine_vector[1], spine_vector[0])))

                if spine_angle > 10:  # Threshold in degrees, adjust as needed
                    issues.append(f"Slouching detected: {spine_angle:.1f}° spine tilt")

        except Exception as e:
            issues.append(f"Error analyzing posture: {str(e)}")

        if not issues:
            issues.append("No posture issues detected")

        return issues

    def run_camera(self, camera_id=0, display=True, record=False, duration=None):
        """
        Run posture detection on camera feed.

        Args:
            camera_id: Camera device ID
            display: Whether to display the feed
            record: Whether to record keypoints
            duration: Optional duration in seconds to run
        """
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

        # Get camera properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:  # Sometimes the fps can't be determined
            fps = 30  # Use a default value

        # Initialize video writer if recording
        video_writer = None
        if record:
            try:
                # First try mp4v codec
                output_video = os.path.join(self.output_dir, f"posture_recording_{int(time.time())}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

                # Test if the video writer is properly initialized
                if not video_writer.isOpened():
                    raise Exception("mp4v codec not available")
            except Exception as e:
                print(f"Warning: Could not use mp4v codec: {e}")
                try:
                    # Fallback to XVID codec
                    output_video = os.path.join(self.output_dir, f"posture_recording_{int(time.time())}.avi")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

                    if not video_writer.isOpened():
                        print("Warning: Could not initialize video recording. Continuing without recording video.")
                        video_writer = None
                except Exception as e2:
                    print(f"Warning: Could not use XVID codec either: {e2}")
                    print("Continuing without video recording.")
                    video_writer = None

        start_time = time.time()
        frame_count = 0
        quality_data = []

        try:
            while True:
                # Check if duration has elapsed
                if duration and (time.time() - start_time) > duration:
                    break

                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break

                # Process frame
                processed_frame, keypoints = self.process_frame(frame)

                # Analyze quality
                quality = self.analyze_quality(keypoints)
                quality_data.append(quality)

                # Detect posture issues
                issues = self.detect_posture_issues(keypoints)

                # Create a copy for overlay to avoid modifying the original frame for recording
                overlay_frame = processed_frame.copy()

                # Add quality metrics and posture issues as text overlay
                cv2.putText(overlay_frame, f"PUK: {quality['PUK']:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(overlay_frame, f"CL: {quality['CL']:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Display posture issues
                for i, issue in enumerate(issues):
                    # Truncate long issues to fit on screen
                    displayed_issue = issue[:50] + "..." if len(issue) > 50 else issue
                    cv2.putText(overlay_frame, displayed_issue, (10, 90 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # For recording use the processed frame, for display use the overlay frame
                display_frame = overlay_frame

                # Save periodic keypoint data if recording
                if record and frame_count % int(fps) == 0:  # Save data once per second
                    self.save_keypoints(keypoints, quality)

                # Record video if enabled
                if video_writer:
                    video_writer.write(processed_frame)  # Record the processed frame without text overlay

                # Display the frame
                if display:
                    cv2.imshow('Posture Detection', display_frame)  # Display the frame with overlay

                    # Exit on 'q' key press or ESC
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 27 is ESC key
                        break

                frame_count += 1

        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            # Release resources
            cap.release()
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()

            # Save quality summary if recording
            if record and quality_data:
                df = pd.DataFrame(quality_data)
                csv_path = os.path.join(self.output_dir, f"quality_summary_{int(time.time())}.csv")
                df.to_csv(csv_path, index=False)
                print(f"Saved quality summary to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Real-time Posture Detection System")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--model_path", type=str, default=None, help="Path to OpenPose models")
    parser.add_argument("--output_dir", type=str, default="posture_data", help="Directory to save output data")
    parser.add_argument("--no_display", action="store_true", help="Don't display the camera feed")
    parser.add_argument("--record", action="store_true", help="Record keypoints and video")
    parser.add_argument("--duration", type=int, help="Duration to run in seconds")

    args = parser.parse_args()

    try:
        print("Initializing Posture Detection System...")
        detector = PostureDetector(args.model_path, args.output_dir)

        print(f"Starting camera feed from camera {args.camera}...")
        detector.run_camera(
            camera_id=args.camera,
            display=not args.no_display,
            record=args.record,
            duration=args.duration
        )

        print("Posture Detection System terminated.")
    except Exception as e:
        print(f"Error running posture detection: {e}")
        print("If MediaPipe is missing, install it with: pip install mediapipe")
        print(
            "For OpenPose, follow installation instructions at: https://github.com/CMU-Perceptual-Computing-Lab/openpose")


if __name__ == "__main__":
    main()