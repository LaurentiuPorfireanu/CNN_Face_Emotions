import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
import time
import queue
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from Helpers.StatusLight import StatusLight

warnings.filterwarnings('ignore')


class PostureDetectorModel:
    def __init__(self, output_dir="posture_data"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(exist_ok=True)

        self.use_mediapipe = False
        self.mp = None
        self.mp_pose = None
        self.mp_drawing = None
        self.pose = None
        self.is_loaded = False

    def load_model(self):
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
            self.is_loaded = True
            return True
        except ImportError:
            print("MediaPipe not found.")
            return False

    def process_frame(self, frame):
        if not self.is_loaded:
            return frame, {'error': 'Model not loaded'}

        if self.use_mediapipe:
            return self._process_with_mediapipe(frame)
        else:
            return frame, {'error': 'No pose detection method available'}

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
                    cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(annotated_frame, label, (x + 10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw lines between keypoints
            if all(label in keypoints["raw"] for label in ["Nose", "Left Shoulder", "Right Shoulder"]):
                nose = keypoints["raw"]["Nose"]
                ls = keypoints["raw"]["Left Shoulder"]
                rs = keypoints["raw"]["Right Shoulder"]

                if all(kp["conf"] > 0.5 for kp in [nose, ls, rs]):
                    # Draw line between shoulders
                    cv2.line(annotated_frame, (ls["x"], ls["y"]), (rs["x"], rs["y"]), (0, 255, 255), 2)

                    # Draw line from nose to shoulder midpoint
                    mid_x = (ls["x"] + rs["x"]) // 2
                    mid_y = (ls["y"] + rs["y"]) // 2
                    cv2.line(annotated_frame, (nose["x"], nose["y"]), (mid_x, mid_y), (255, 0, 0), 2)

                    # Draw a vertical reference line from midpoint of shoulders
                    top_y = min(nose["y"], ls["y"], rs["y"]) - 50
                    cv2.line(annotated_frame, (mid_x, mid_y), (mid_x, top_y), (0, 0, 255), 2)

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
                return "Insufficient data", 0.0
            nose = keypoints["raw"].get("Nose")
            ls = keypoints["raw"].get("Left Shoulder")
            rs = keypoints["raw"].get("Right Shoulder")
            if not nose or not ls or not rs:
                return "Missing key landmarks", 0.0
            if nose["conf"] < 0.5 or ls["conf"] < 0.5 or rs["conf"] < 0.5:
                return "Low confidence in landmarks", 0.0

            # Calculate shoulder midpoint y-position
            shoulder_mid_y = (ls["y"] + rs["y"]) / 2

            # Calculate shoulder width for scaling
            shoulder_width = abs(rs["x"] - ls["x"])

            # Calculate vertical distance from shoulders to nose
            # Positive value: nose is higher than shoulders (good)
            vertical_distance = shoulder_mid_y - nose["y"]

            # Normalize by shoulder width to account for different body sizes and distances from camera
            vertical_ratio = vertical_distance / shoulder_width if shoulder_width > 0 else 0

            # Return the vertical ratio and text status
            offset = 0.1
            if vertical_ratio < 0.4 - offset:
                return "Severe forward head tilt", 1.0 - vertical_ratio  # High tilt value
            elif vertical_ratio < 0.55 - offset:
                return "Moderate forward head tilt", 0.7 - vertical_ratio  # Medium tilt value
            elif vertical_ratio < 0.7 - offset:
                return "Slight forward head tilt", 0.4 - vertical_ratio  # Low tilt value
            else:
                return "Good posture", 0.0  # No tilt

        except Exception as e:
            return f"Error: {e}", 0.0

    def get_default_output(self):
        return "neutral", 0.0, {'arousal': 0.0, 'valence': 0.0, 'dominance': 0.0}


class SimplePostureWidget(ttk.Frame):
	def __init__(self, parent, width=300, height=200, **kwargs):
		ttk.Frame.__init__(self, parent, **kwargs)

		self.parent = parent
		self.width = width
		self.height = height

		self.model = PostureDetectorModel()
		self.is_active = False
		self.is_loading = False
		self.current_posture = "Unknown"
		self.current_tilt_value = 0.0

		self.title_label = None
		self.result_queue = queue.Queue()

		self.create_ui()
		self.check_results()

	def create_ui(self):
		self.grid_propagate(False)
		self.config(width=self.width, height=self.height)

		self.grid_columnconfigure(0, weight=1)
		self.grid_rowconfigure(0, weight=0)
		self.grid_rowconfigure(1, weight=1)
		self.grid_rowconfigure(2, weight=0)

		title_frame = ttk.Frame(self)
		title_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 0))

		self.title_label = ttk.Label(
			title_frame,
			text="Head Tilt Level",
			font=("Arial", 12, "bold"),
			anchor="center"
		)
		self.title_label.pack(fill="x")

		chart_frame = ttk.Frame(self)
		chart_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=2)

		self.create_chart_frame(chart_frame)

		button_frame = ttk.Frame(self)
		button_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=(0, 2))

		button_frame.columnconfigure(0, weight=1)
		button_frame.columnconfigure(1, weight=1)
		button_frame.columnconfigure(2, weight=0)

		self.load_button = ttk.Button(
			button_frame,
			text="Load Model",
			command=self.toggle_model
		)
		self.load_button.grid(row=0, column=0, sticky="ew", padx=(0, 2))

		self.start_button = ttk.Button(
			button_frame,
			text="Start",
			command=self.toggle_processing,
			state="disabled"
		)
		self.start_button.grid(row=0, column=1, sticky="ew", padx=(2, 2))

		self.status_light = StatusLight(button_frame, size=15)
		self.status_light.grid(row=0, column=2, padx=(2, 0), pady=0, sticky="e")
		self.status_light.set_state("off")

	def create_chart_frame(self, parent):
		parent.columnconfigure(0, weight=1)
		parent.rowconfigure(0, weight=1)

		is_dark_theme = self.is_dark_theme()
		bg_color = '#2b2b2b' if is_dark_theme else 'white'
		text_color = 'white' if is_dark_theme else 'black'

		self.fig = Figure(figsize=(self.width / 100, (self.height * 0.85) / 100), dpi=100)
		self.fig.patch.set_facecolor(bg_color)
		self.fig.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1)

		self.ax = self.fig.add_subplot(111)
		self.ax.set_ylim(0, 1)
		self.ax.set_xlim(-1, 1)
		self.ax.set_xticks([])

		self.bar = self.ax.bar([0], [0], width=0.5, color='green')[0]

		
		self.ax.set_facecolor(bg_color)
		for spine in self.ax.spines.values():
			spine.set_color(text_color)
		self.ax.tick_params(axis='y', colors=text_color, labelsize=8)

		self.canvas = FigureCanvasTkAgg(self.fig, parent)
		self.canvas.draw()
		canvas_widget = self.canvas.get_tk_widget()
		canvas_widget.grid(row=0, column=0, sticky="nsew")

	def is_dark_theme(self):
		try:
			style = ttk.Style()
			bg_color = style.lookup('TFrame', 'background')
			if not bg_color:
				tmp = ttk.Frame(self)
				bg_color = tmp.winfo_rgb(tmp.cget('background'))
				tmp.destroy()
				brightness = (bg_color[0] + bg_color[1] + bg_color[2]) / (3 * 257)
				return brightness < 128
			else:
				if isinstance(bg_color, str) and bg_color.startswith('#'):
					bg_color = tuple(int(bg_color[i:i + 2], 16) for i in (1, 3, 5))
				else:
					tmp = ttk.Frame(self)
					bg_color = tmp.winfo_rgb(bg_color)
					tmp.destroy()
			brightness = sum(bg_color) / (3 * 257)
			return brightness < 128
		except:
			return False

	def toggle_model(self):
		if self.is_loading:
			return
		if self.model.is_loaded and self.is_active:
			return

		if not self.model.is_loaded:
			self.is_loading = True
			self.load_button.config(state="disabled")
			self.status_light.set_state("loading")

			def load_model_thread():
				success = self.model.load_model()
				if success:
					self.result_queue.put(("model_loaded", None))
				else:
					self.result_queue.put(("model_load_failed", None))
				self.is_loading = False

			threading.Thread(target=load_model_thread).start()
		else:
			self.model.is_loaded = False
			self.load_button.config(text="Load Model")
			self.start_button.config(state="disabled")
			self.status_light.set_state("off")
			self.clear_chart()
			self.title_label.config(text="Head Tilt Level")

	def toggle_processing(self):
		if not self.model.is_loaded or self.is_loading:
			return

		if not self.is_active:
			self.is_active = True
			self.start_button.config(text="Stop")
			self.status_light.set_state("active")
		else:
			self.is_active = False
			self.start_button.config(text="Start")
			self.status_light.set_state("ready")
			self.clear_chart()
			self.title_label.config(text="Head Tilt Level")

	def process_frame(self, frame):
		if not self.is_active or not self.model.is_loaded:
			return frame

		processed_frame, keypoints = self.model.process_frame(frame)
		quality = self.model.analyze_quality(keypoints)
		posture_status, tilt_value = self.model.detect_slouching(keypoints)

		self.result_queue.put(("posture_update", (posture_status, tilt_value)))
		return processed_frame

	def check_results(self):
		try:
			while True:
				msg_type, msg_data = self.result_queue.get_nowait()

				if msg_type == "model_loaded":
					self.load_button.config(text="Unload", state="normal")
					self.start_button.config(state="normal")
					self.status_light.set_state("ready")

				elif msg_type == "model_load_failed":
					self.load_button.config(text="Load", state="normal")
					self.status_light.set_state("off")
					self.update_chart("Error: Failed to load model", 0.0)

				elif msg_type == "posture_update":
					posture_status, tilt_value = msg_data
					self.update_chart(posture_status, tilt_value)

		except queue.Empty:
			pass

		self.after(50, self.check_results)

	def update_chart(self, status_text, tilt_value):
		self.title_label.config(text=status_text)
		tilt_value = max(tilt_value, 0.05)
		self.bar.set_height(tilt_value)

		status_to_color = {
		    "Good posture": "green",
		    "Slight forward head tilt": "yellow",
		    "Moderate forward head tilt": "orange",
		    "Severe forward head tilt": "red"
	    }
		bar_color = status_to_color.get(status_text.strip(), "green")

		self.bar.set_color(bar_color)
		self.canvas.draw()

	def clear_chart(self):
		self.bar.set_height(0)
		self.bar.set_color('green')
		self.title_label.config(text="Head Tilt Level")
		self.canvas.draw()

if __name__ == "__main__":
    # Example usage in a parent window
    root = tk.Tk()
    root.title("Posture Widget Demo")
    root.geometry("200x200")

    posture_widget = SimplePostureWidget(root, width=200, height=200)
    posture_widget.pack(fill=tk.BOTH, expand=True)

    # In a real implementation, you would capture video in the parent
    # and pass frames to posture_widget.process_frame()

    root.mainloop()