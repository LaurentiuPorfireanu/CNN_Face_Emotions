import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
import time
import queue
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from Helpers.StatusLight import StatusLight

warnings.filterwarnings('ignore')


class EmotionCNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) for emotion classification from grayscale facial images.
    """

    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.image_size = 96

        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.2)

        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.2)

        # Third convolutional block
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.2)

        # Compute the number of flattened features dynamically
        self.flat_features = 512 * (self.image_size // 8) * (self.image_size // 8)

        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_features, 512)
        self.dropout4 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.dropout5 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Apply first convolutional block
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Apply second convolutional block
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Apply third convolutional block
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flatten the tensor for fully connected layers
        x = x.view(-1, self.flat_features)

        # Pass through fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = F.relu(self.fc2(x))
        x = self.dropout5(x)
        x = self.fc3(x)  # Final layer outputs raw logits

        return x


class FaceEmotionDetectorModel:
    def __init__(self):
        self.coarse_emotions = ['happy', 'fear', 'angry', 'other']
        self.fine_emotions = ['neutral', 'disgust', 'sad', 'surprise']
        self.all_emotions = self.coarse_emotions[:3] + self.fine_emotions
        self.image_size = 96
        self.threshold = 0.6
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_coarse = None
        self.model_fine = None
        self.face_cascade = None
        self.transform = None
        self.is_loaded = False

    def load_model(self):
        try:
            # Load face detection cascade
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            # Set up image transformation
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

            # Load emotion models - CHANGE THE PATHS HERE
            self.model_coarse = EmotionCNN(num_classes=len(self.coarse_emotions)).to(self.device)
            self.model_fine = EmotionCNN(num_classes=len(self.fine_emotions)).to(self.device)

            # Use os.path.join to create proper paths to the model files
            model_dir = os.path.dirname(os.path.abspath(__file__))
            coarse_model_path = os.path.join(model_dir, 'coarse_model.pth')
            fine_model_path = os.path.join(model_dir, 'fine_model.pth')

            self.model_coarse.load_state_dict(torch.load(coarse_model_path, map_location=self.device))
            self.model_fine.load_state_dict(torch.load(fine_model_path, map_location=self.device))

            self.model_coarse.eval()
            self.model_fine.eval()

            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def process_frame(self, frame):
        if not self.is_loaded:
            return frame, {'error': 'Model not loaded'}

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

            if len(faces) == 0:
                return frame, {'error': 'No face detected'}

            # Get the first face
            x, y, w, h = faces[0]

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract and process face
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (self.image_size, self.image_size))
            face_pil = Image.fromarray(face)
            tensor = self.transform(face_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # Run through coarse model first
                out_c = self.model_coarse(tensor)
                probs_c = F.softmax(out_c, dim=1)[0]
                p_other = probs_c[self.coarse_emotions.index('other')]

                if p_other > self.threshold:
                    # Use fine model if 'other' has high probability
                    out_f = self.model_fine(tensor)
                    probs_f = F.softmax(out_f, dim=1)[0]
                    idx_f = torch.argmax(out_f, dim=1).item()
                    emotion = self.fine_emotions[idx_f]
                    confidence = probs_f[idx_f].item()

                    # Create combined array for visualization
                    all_probs = torch.zeros(len(self.all_emotions))
                    for i, e in enumerate(self.fine_emotions):
                        idx = self.all_emotions.index(e)
                        all_probs[idx] = probs_f[i]
                else:
                    # Use coarse model results
                    top3 = probs_c[:3]  # Exclude 'other'
                    idx_c = torch.argmax(top3).item()
                    emotion = self.coarse_emotions[idx_c]
                    confidence = top3[idx_c].item()

                    # Create combined array for visualization
                    all_probs = torch.zeros(len(self.all_emotions))
                    for i, e in enumerate(self.coarse_emotions[:3]):  # Exclude 'other'
                        idx = self.all_emotions.index(e)
                        all_probs[idx] = probs_c[i]

            result = {
                'emotion': emotion,
                'confidence': confidence,
                'face_location': (x, y, w, h),
                'probabilities': all_probs.cpu().numpy()
            }

            return frame, result

        except Exception as e:
            print(f"Error in processing: {e}")
            return frame, {'error': f'Processing error: {str(e)}'}

    def get_default_output(self):
        return {'error': 'Model not loaded or processing failed'}


class SimpleFaceEmotionWidget(ttk.Frame):
    def __init__(self, parent, width=500, height=200, **kwargs):
        ttk.Frame.__init__(self, parent, **kwargs)

        self.parent = parent
        self.width = width
        self.height = height

        self.model = FaceEmotionDetectorModel()

        self.is_loading = False
        self.is_active = False
        self.frame_count = 0
        self.current_emotion = "unknown"
        self.current_confidence = 0.0
        self.last_update_time = time.time()

        self.result_queue = queue.Queue()

        self.create_ui()
        self.check_results()

    def create_ui(self):
        self.grid_propagate(False)
        self.config(width=self.width, height=self.height)

        # Configure grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)  # Chart area
        self.grid_rowconfigure(1, weight=0)  # Control panel with buttons

        # Chart frame - takes most of the space
        chart_frame = ttk.Frame(self)
        chart_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.create_chart_frame(chart_frame)

        # Control panel at bottom - modified to put buttons on the left
        control_frame = ttk.Frame(self)
        control_frame.grid(row=1, column=0, sticky="ew", padx=2, pady=2)

        # Modified grid configuration - all controls at left, with empty space to the right
        control_frame.grid_columnconfigure(0, weight=0)  # Load button
        control_frame.grid_columnconfigure(1, weight=0)  # Start button
        control_frame.grid_columnconfigure(2, weight=0)  # Status light
        control_frame.grid_columnconfigure(3, weight=1)  # Empty space takes remaining width

        # Move all controls to the left side
        self.load_button = ttk.Button(
            control_frame,
            text="Load Model",
            command=self.toggle_model,
            width=12
        )
        self.load_button.grid(row=0, column=0, padx=(2, 2), pady=2, sticky="w")

        self.start_button = ttk.Button(
            control_frame,
            text="Start",
            command=self.toggle_processing,
            state="disabled",
            width=12
        )
        self.start_button.grid(row=0, column=1, padx=(2, 2), pady=2, sticky="w")

        self.status_light = StatusLight(control_frame, size=15)
        self.status_light.grid(row=0, column=2, padx=(2, 5), pady=2, sticky="w")
        self.status_light.set_state("off")

    def create_chart_frame(self, parent):
        """Create the chart frame for emotion visualization"""
        # Configure the parent frame
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        is_dark_theme = self.is_dark_theme()
        bg_color = '#2b2b2b' if is_dark_theme else 'white'
        text_color = 'white' if is_dark_theme else 'black'
        grid_color = '#404040' if is_dark_theme else '#cccccc'

        # Create matplotlib figure
        self.fig = Figure(figsize=(self.width / 100, (self.height * 0.7) / 100), dpi=100)
        self.fig.patch.set_facecolor(bg_color)

        # Add proper padding
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.15)

        # Create subplot for the bar chart
        self.ax = self.fig.add_subplot(111)

        # Get emotion labels
        self.emotion_labels = ['happy', 'fear', 'angry', 'neutral', 'disgust', 'sad', 'surprise']
        x_pos = np.arange(len(self.emotion_labels))

        # Create initial empty bars
        self.bars = self.ax.bar(x_pos, np.zeros(len(self.emotion_labels)),
                                width=0.7,
                                color=['#FF9999', '#FFCC99', '#FF6666', '#99CCFF', '#CC99FF', '#9999FF', '#CCFF99'])

        # Set x-axis labels
        self.ax.set_xticks(x_pos)
        self.ax.set_xticklabels(self.emotion_labels, fontsize=8)
        self.ax.set_ylim(0, 1)
        self.ax.grid(True, alpha=0.3, color=grid_color)

        # Setup canvas appearance
        self.ax.set_facecolor(bg_color)
        for spine in self.ax.spines.values():
            spine.set_color(text_color)

        self.ax.tick_params(axis='both', colors=text_color, labelsize=8)

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky="nsew", pady=(0, 5))

    def is_dark_theme(self):
        """Detect if dark theme is being used"""
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
                if isinstance(bg_color, str):
                    if bg_color.startswith('#'):
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
        """Toggle model loading/unloading"""
        if self.is_loading:
            return

        # Add this check to prevent unloading while active
        if self.model.is_loaded and self.is_active:
            return  # Don't allow unloading when the widget is active

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

    def toggle_processing(self):
        """Toggle face processing on/off"""
        if not self.model.is_loaded or self.is_loading:
            return

        if not self.is_active:
            self.is_active = True
            self.start_button.config(text="Stop")
            self.status_light.set_state("active")
            self.frame_count = 0
        else:
            self.is_active = False
            self.start_button.config(text="Start")
            self.status_light.set_state("ready")
            self.clear_chart()

    def process_frame(self, frame):
        """Process a video frame (called from parent widget)"""
        if not self.is_active or not self.model.is_loaded:
            return frame

        # Update frame counter (internally only)
        self.frame_count = (self.frame_count + 1) % 15

        # Throttle processing to avoid overloading CPU
        current_time = time.time()
        if current_time - self.last_update_time > 0.2:  # Process at most every 200ms
            # Process frame in a separate thread
            threading.Thread(
                target=self.process_frame_thread,
                args=(frame.copy(),)
            ).start()
            self.last_update_time = current_time

        # Return the original frame without modification
        # The UI doesn't show the camera view directly
        return frame

    def process_frame_thread(self, frame):
        """Process frame in a separate thread"""
        try:
            # Use the model to process the frame
            _, result = self.model.process_frame(frame)
            if 'error' not in result:
                self.result_queue.put(("emotion_result", result))
        except Exception as e:
            print(f"Error processing frame: {e}")

    def check_results(self):
        """Check and process results from the result queue"""
        try:
            while True:
                msg_type, msg_data = self.result_queue.get_nowait()

                if msg_type == "model_loaded":
                    self.load_button.config(text="Unload Model", state="normal")
                    self.start_button.config(state="normal")
                    self.status_light.set_state("ready")

                elif msg_type == "model_load_failed":
                    self.load_button.config(text="Load Model", state="normal")
                    self.status_light.set_state("off")

                elif msg_type == "emotion_result":
                    emotion = msg_data['emotion']
                    confidence = msg_data['confidence']
                    probabilities = msg_data['probabilities']
                    self.update_emotion_display(emotion, confidence, probabilities)

        except queue.Empty:
            pass

        self.after(50, self.check_results)

    def update_emotion_display(self, emotion, confidence, probabilities):
        """Update the emotion visualization"""
        self.current_emotion = emotion
        self.current_confidence = confidence

        # Update bars
        for i, bar in enumerate(self.bars):
            if i < len(probabilities):
                bar.set_height(probabilities[i])
            else:
                bar.set_height(0)

        # Highlight the current emotion bar
        if emotion in self.emotion_labels:
            idx = self.emotion_labels.index(emotion)
            for i, bar in enumerate(self.bars):
                bar.set_alpha(1.0 if i == idx else 0.7)

        # Redraw canvas
        self.canvas.draw()

    def clear_chart(self):
        """Reset the chart to default state"""
        # Reset all bars to zero
        for bar in self.bars:
            bar.set_height(0)
            bar.set_alpha(0.7)

        # Reset frame counter
        self.frame_count = 0

        # Redraw canvas
        self.canvas.draw()


if __name__ == "__main__":
    # Test window
    root = tk.Tk()
    root.title("Face Emotion Detection")
    root.geometry("500x200")

    face_emotion_widget = SimpleFaceEmotionWidget(root, width=500, height=200)
    face_emotion_widget.pack(fill=tk.BOTH, expand=True)

    # In a real implementation, you would capture video in the parent app
    # and pass frames to face_emotion_widget.process_frame()

    root.mainloop()