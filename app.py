import tkinter as tk
from tkinter import ttk
import cv2
import threading
import time
import numpy as np
import warnings
import PIL.Image, PIL.ImageTk
from pathlib import Path
import os

# Import the component widgets
# Note: These imports assume the widget scripts are in their appropriate directories
from VoiceAndSpeech.voice import VoiceEmotionDetector
from VoiceAndSpeech.speech import SpeechRecognitionWidget
from BodyPosture.posture import SimplePostureWidget
from FaceEmotion.face import SimpleFaceEmotionWidget  # Add this import for the face widget

warnings.filterwarnings('ignore')


class MultimodalAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multimodal Analysis System")

        # Get screen dimensions and set up window size
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{sw}x{sh}+0+0")

        # Create folder for saved data if it doesn't exist
        self.data_dir = Path("multimodal_data")
        self.data_dir.mkdir(exist_ok=True)

        # Initialize camera variables
        self.cap = cv2.VideoCapture(0)
        self.is_camera_active = True
        self.current_frame = None
        self.processed_frame = None
        self.video_after_id = None
        self.running = True

        # Widget dimensions
        self.face_width = 500
        self.face_height = 200
        self.voice_width = 300
        self.voice_height = 200
        self.posture_width = 200
        self.posture_height = 200
        self.speech_width = 500
        self.speech_height = 200

        # Combined widget width (for right side)
        self.widget_container_width = max(self.voice_width + self.posture_width, self.speech_width, self.face_width)

        # Quit on 'q'
        self.root.bind('<KeyPress-q>', lambda e: self.on_close())

        # Build the UI
        self.build_ui()

        # Start the update loop for the camera feed
        self.video_after_id = self.root.after(10, self.video_loop)

    def build_ui(self):
        """Create the main UI layout"""
        # Configure dark theme colors
        bg_color = "#2c2c2c"
        frame_color = "#3a3a3a"
        text_color = "white"

        self.root.configure(bg=bg_color)

        # Main frame
        main_frame = tk.Frame(self.root, bg=bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a PanedWindow to separate camera from widgets
        paned = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL, bg=bg_color, sashwidth=5)
        paned.pack(fill=tk.BOTH, expand=True)

        # Camera frame - large, fills most of the left side
        camera_frame = tk.LabelFrame(
            paned,
            text="Camera Feed",
            bg=frame_color,
            fg=text_color,
            font=("Helvetica", 12, "bold")
        )

        # Widget container - right side
        widget_container = tk.Frame(paned, bg=bg_color, width=self.widget_container_width)

        # Add frames to paned window
        paned.add(camera_frame, stretch="always")
        paned.add(widget_container, stretch="never")

        # Camera canvas
        self.canvas = tk.Canvas(camera_frame, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Configure the widget container grid - now with 3 rows
        widget_container.grid_columnconfigure(0, weight=1)  # Posture column
        widget_container.grid_columnconfigure(1, weight=3)  # Voice column
        widget_container.grid_rowconfigure(0, weight=1)  # Top row - Face
        widget_container.grid_rowconfigure(1, weight=1)  # Middle row - Posture & Voice
        widget_container.grid_rowconfigure(2, weight=1)  # Bottom row - Speech

        # Top row - Face widget (spans both columns)
        face_frame = tk.LabelFrame(
            widget_container,
            text="Face Emotion Detection",
            bg=frame_color,
            fg=text_color,
            font=("Helvetica", 12, "bold"),
            width=self.face_width,
            height=self.face_height
        )
        face_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=2, pady=2)
        face_frame.grid_propagate(False)  # Keep fixed size

        # Face widget - use the dimensions specified
        self.face_widget = SimpleFaceEmotionWidget(
            face_frame,
            width=self.face_width,
            height=self.face_height
        )
        self.face_widget.pack(fill=tk.BOTH, expand=True)

        # Middle row - Posture widget (left)
        posture_frame = tk.LabelFrame(
            widget_container,
            text="Posture Analysis",
            bg=frame_color,
            fg=text_color,
            font=("Helvetica", 12, "bold"),
            width=self.posture_width,
            height=self.posture_height
        )
        posture_frame.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        posture_frame.grid_propagate(False)  # Keep fixed size

        # Posture widget - use the dimensions specified
        self.posture_widget = SimplePostureWidget(
            posture_frame,
            width=self.posture_width,
            height=self.posture_height
        )
        self.posture_widget.pack(fill=tk.BOTH, expand=True)

        # Middle row - Voice widget (right)
        voice_frame = tk.LabelFrame(
            widget_container,
            text="Voice Emotion",
            bg=frame_color,
            fg=text_color,
            font=("Helvetica", 12, "bold"),
            width=self.voice_width,
            height=self.voice_height
        )
        voice_frame.grid(row=1, column=1, sticky="nsew", padx=2, pady=2)
        voice_frame.grid_propagate(False)  # Keep fixed size

        # Voice widget - use the dimensions specified
        self.voice_widget = VoiceEmotionDetector(
            voice_frame,
            width=self.voice_width,
            height=self.voice_height
        )
        self.voice_widget.pack(fill=tk.BOTH, expand=True)

        # Bottom row - Speech widget (spans both columns)
        speech_frame = tk.LabelFrame(
            widget_container,
            text="Speech Recognition",
            bg=frame_color,
            fg=text_color,
            font=("Helvetica", 12, "bold"),
            width=self.speech_width,
            height=self.speech_height
        )
        speech_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=2, pady=2)
        speech_frame.grid_propagate(False)  # Keep fixed size

        # Speech widget - use the dimensions specified
        self.speech_widget = SpeechRecognitionWidget(
            speech_frame,
            width=self.speech_width,
            height=self.speech_height
        )
        self.speech_widget.pack(fill=tk.BOTH, expand=True)

        # Set paned window position - give more space to camera
        camera_width = self.root.winfo_screenwidth() - self.widget_container_width - 50
        paned.paneconfigure(camera_frame, minsize=camera_width)
        paned.paneconfigure(widget_container, minsize=self.widget_container_width)

    def video_loop(self):
        """Update the camera feed on the canvas"""
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            time.sleep(0.5)
            self.cap = cv2.VideoCapture(0)
            self.video_after_id = self.root.after(30, self.video_loop)
            return

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        self.current_frame = frame.copy()

        # Process frame for face emotion detection if active
        if hasattr(self.face_widget, 'is_active') and self.face_widget.is_active:
            # Process the frame but don't replace the display frame yet
            self.face_widget.process_frame(frame)

        # Process frame for posture detection if active
        if hasattr(self.posture_widget, 'is_active') and self.posture_widget.is_active:
            self.processed_frame = self.posture_widget.process_frame(frame)
            display_frame = self.processed_frame
        else:
            display_frame = frame

        # Convert to RGB for display in Tkinter
        img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(img)

        # Resize to fit canvas while maintaining aspect ratio
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            # Calculate scaling to fit in canvas
            img_width, img_height = img.size
            scale = min(canvas_width / img_width, canvas_height / img_height)

            new_size = (int(img_width * scale), int(img_height * scale))
            img = img.resize(new_size, PIL.Image.LANCZOS)

        # Display the image
        self.photo = PIL.ImageTk.PhotoImage(image=img)

        # Clear canvas and display new image
        self.canvas.delete("all")
        x = (canvas_width - img.width) // 2
        y = (canvas_height - img.height) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)

        # Schedule the next update
        self.video_after_id = self.root.after(33, self.video_loop)  # ~30fps

    def on_close(self):
        """Handle window closing"""
        if self.video_after_id:
            self.root.after_cancel(self.video_after_id)
        self.running = False

        # Stop each analysis module if active
        if hasattr(self.face_widget, 'is_active') and self.face_widget.is_active:
            self.face_widget.toggle_processing()

        if hasattr(self.voice_widget, 'is_recording') and self.voice_widget.is_recording:
            self.voice_widget.toggle_recording()

        if hasattr(self.speech_widget, 'is_recording') and self.speech_widget.is_recording:
            self.speech_widget.toggle_recording()

        if hasattr(self.posture_widget, 'is_active') and self.posture_widget.is_active:
            self.posture_widget.toggle_processing()

        # Release camera
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

        # Destroy window
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = MultimodalAnalysisApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()