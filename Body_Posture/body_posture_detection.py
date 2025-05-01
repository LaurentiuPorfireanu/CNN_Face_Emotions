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
        self.output_dir = output_dir
        Path(output_dir).mkdir(exist_ok=True)

        self.use_mediapipe = False
        self.op = None

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
        pass  # Not needed for this use-case

    def process_frame(self, frame):
        if self.use_mediapipe:
            return self._process_with_mediapipe(frame)
        else:
            return frame, {"error": "No pose detection method available"}

    def _process_with_mediapipe(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        annotated_frame = frame.copy()

        keypoint_indices = {
            0: "Nose",
            11: "Left Shoulder", 12: "Right Shoulder"
        }

        if results.pose_landmarks:
            h, w, _ = frame.shape
            keypoints = {"x": [], "y": [], "confidence": [], "labels": [], "raw": {}}
            for idx, label in keypoint_indices.items():
                landmark = results.pose_landmarks.landmark[idx]
                x, y = int(landmark.x * w), int(landmark.y * h)
                conf = landmark.visibility
                keypoints["x"].append(x)
                keypoints["y"].append(y)
                keypoints["confidence"].append(conf)
                keypoints["labels"].append(label)
                keypoints["raw"][label] = {"x": x, "y": y, "conf": conf}

                if conf > 0.5:
                    cv2.circle(annotated_frame, (x, y), 2, (0, 255, 0), -1)
            return annotated_frame, keypoints
        else:
            return annotated_frame, {"error": "No face/shoulder keypoints detected"}

    def analyze_quality(self, keypoints):
        if "error" in keypoints:
            return {"PUK": 1.0, "CL": 0.0}
        total_points = len(keypoints["x"])
        unrecognized = sum(1 for i in range(total_points)
                           if keypoints["x"][i] == 0 and keypoints["y"][i] == 0 and keypoints["confidence"][i] == 0)
        puk = unrecognized / total_points if total_points > 0 else 1.0
        cl = np.mean(keypoints["confidence"]) if keypoints["confidence"] else 0.0
        return {"PUK": puk, "CL": cl}

    def detect_slouching(self, keypoints):
        try:
            if "raw" not in keypoints:
                return "Insufficient data"
            nose = keypoints["raw"].get("Nose")
            ls = keypoints["raw"].get("Left Shoulder")
            rs = keypoints["raw"].get("Right Shoulder")
            if not nose or not ls or not rs:
                return "Missing key landmarks"
            if nose["conf"] < 0.5 or ls["conf"] < 0.5 or rs["conf"] < 0.5:
                return "Low confidence in landmarks"

            # Calculate shoulder midpoint y-position
            shoulder_mid_y = (ls["y"] + rs["y"]) / 2

            # Calculate shoulder width for scaling
            shoulder_width = abs(rs["x"] - ls["x"])

            # Calculate vertical distance from shoulders to nose
            # Positive value: nose is higher than shoulders (good)
            vertical_distance = shoulder_mid_y - nose["y"]

            # Normalize by shoulder width to account for different body sizes and distances from camera
            vertical_ratio = vertical_distance / shoulder_width if shoulder_width > 0 else 0

            # Draw a vertical reference line from shoulders to help visualize
            # This could be implemented in the visualization part of the code
            offset = 0.1
            # Determine posture status based on the vertical ratio
            # These thresholds should be adjusted based on testing with your camera setup
            if vertical_ratio < 0.4 - offset:
                return f"Severe forward head tilt (ratio: {vertical_ratio:.2f})"
            elif vertical_ratio < 0.55 - offset:
                return f"Moderate forward head tilt (ratio: {vertical_ratio:.2f})"
            elif vertical_ratio < 0.7 - offset:
                return f"Slight forward head tilt (ratio: {vertical_ratio:.2f})"
            else:
                return f"Good posture (ratio: {vertical_ratio:.2f})"

        except Exception as e:
            return f"Error: {e}"

    def save_keypoints(self, keypoints, quality, filename=None):
        if "error" in keypoints:
            print(f"Not saving data: {keypoints['error']}")
            return
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "keypoints": keypoints,
            "quality": quality
        }
        if filename is None:
            filename = f"face_shoulders_{int(time.time())}.json"
        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved data to {file_path}")

    def run_camera(self, camera_id=0, display=True, record=False, duration=None):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30

        video_writer = None
        if record:
            output_video = os.path.join(self.output_dir, f"face_shoulders_{int(time.time())}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

        start_time = time.time()
        frame_count = 0

        try:
            while True:
                if duration and (time.time() - start_time) > duration:
                    break

                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break

                processed_frame, keypoints = self.process_frame(frame)
                quality = self.analyze_quality(keypoints)
                slouch_status = self.detect_slouching(keypoints)

                overlay_frame = processed_frame.copy()
                cv2.putText(overlay_frame, f"PUK: {quality['PUK']:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(overlay_frame, f"CL: {quality['CL']:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(overlay_frame, f"{slouch_status}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                for i, label in enumerate(keypoints.get("labels", [])):
                    x = keypoints["x"][i]
                    y = keypoints["y"][i]
                    cv2.putText(overlay_frame, f"{label}: ({x}, {y})", (10, 120 + i * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                if record and frame_count % int(fps) == 0:
                    self.save_keypoints(keypoints, quality)

                if video_writer:
                    video_writer.write(processed_frame)

                if display:
                    cv2.imshow('Face & Shoulders Detection', overlay_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        break

                frame_count += 1

        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Face & Shoulders Detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--output_dir", type=str, default="posture_data", help="Directory to save output data")
    parser.add_argument("--no_display", action="store_true", help="Don't display the camera feed")
    parser.add_argument("--record", action="store_true", help="Record keypoints and video")
    parser.add_argument("--duration", type=int, help="Duration to run in seconds")

    args = parser.parse_args()

    detector = PostureDetector(output_dir=args.output_dir)
    detector.run_camera(
        camera_id=args.camera,
        display=not args.no_display,
        record=args.record,
        duration=args.duration
    )


if __name__ == "__main__":
    main()