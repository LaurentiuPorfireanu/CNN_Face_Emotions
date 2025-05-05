import os
import importlib.util
import threading
import time

import tkinter as tk
import cv2
from PIL import Image, ImageTk

import torch
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Helper functions ---------------------------------------------------------

def find_file(root: str, filename: str) -> str:
    """
    Recursively search for a file under the given root directory.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    raise FileNotFoundError(f"Could not find {filename} under {root}")

def load_module_from_path(name: str, path: str):
    """
    Dynamically load a Python module from a file path.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# --- Project setup ------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)

# Find the required scripts in the project
EMOTION_SCRIPT = find_file(ROOT, 'realtimedetection.py')
POSTURE_SCRIPT = find_file(ROOT, 'body_posture_detection.py')
FIDGET_SCRIPT  = find_file(ROOT, 'main.py')

realtime_mod = load_module_from_path('emotions', EMOTION_SCRIPT)
posture_mod  = load_module_from_path('posture',  POSTURE_SCRIPT)
fidget_mod   = load_module_from_path('fidget',   FIDGET_SCRIPT)

# --- Emotion model loading ---------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EmotionCNN      = realtime_mod.EmotionCNN
COARSE_EMOTIONS = realtime_mod.COARSE_EMOTIONS
FINE_EMOTIONS   = realtime_mod.FINE_EMOTIONS
IMAGE_SIZE      = realtime_mod.IMAGE_SIZE

# Initialize and load pretrained weights
model_coarse = EmotionCNN(num_classes=len(COARSE_EMOTIONS)).to(device)
model_fine   = EmotionCNN(num_classes=len(FINE_EMOTIONS)).to(device)

model_coarse.load_state_dict(
    torch.load(os.path.join(ROOT, 'coarse_model.pth'), map_location=device)
)
model_fine.load_state_dict(
    torch.load(os.path.join(ROOT, 'fine_model.pth'), map_location=device)
)

model_coarse.eval()
model_fine.eval()

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Transform pipeline for emotion detection
emo_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def detect_emotion_probs(frame, threshold: float = 0.4):
    """
    Detect face in the frame, run the coarse model,
    and if needed the fine model, returning a dict of emotion probabilities.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    if len(faces) == 0:
        return {e: 0.0 for e in COARSE_EMOTIONS + FINE_EMOTIONS}

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
    face = Image.fromarray(face)
    tensor = emo_transform(face).unsqueeze(0).to(device)

    probs = {}
    with torch.no_grad():
        out_c = model_coarse(tensor)
        pc   = F.softmax(out_c, dim=1)[0].cpu().numpy()
        for i, emo in enumerate(COARSE_EMOTIONS):
            probs[emo] = float(pc[i])
        # If 'other' has high confidence, run fine-grained model
        if pc[COARSE_EMOTIONS.index('other')] > threshold:
            out_f = model_fine(tensor)
            pf    = F.softmax(out_f, dim=1)[0].cpu().numpy()
            for i, emo in enumerate(FINE_EMOTIONS):
                probs[emo] = float(pf[i])
    return probs

# --- Posture & Fidget detectors ----------------------------------------------

PostureDetector    = posture_mod.PostureDetector
HandFidgetDetector = fidget_mod.HandFidgetDetector
HandState          = fidget_mod.HandState

# Colors for emotion bars
EMO_BAR_COLORS  = {
    "happy":"#FFD700","fear":"#FF8C00","angry":"#FF4500",
    "neutral":"#808080","disgust":"#800080","sad":"#1E90FF","surprise":"#32CD32"
}
# Colors for posture bars
POST_BAR_COLORS = ["#2E8B57","#FFD700","#FF8C00","#FF4500"]

# --- Main Application --------------------------------------------------------

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Live Monitor")

        # Set window to full screen resolution (but not fullscreen mode)
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_w}x{screen_h}+0+0")

        # Bind 'q' key to quit
        self.root.bind('<KeyPress-q>', lambda e: self.on_close())
        self.running = True

        # Video capture setup
        self.cap            = cv2.VideoCapture(0)
        self.last_frame     = None
        self.video_after_id = None

        # Data holders
        self.emo_labels    = [e for e in COARSE_EMOTIONS + FINE_EMOTIONS if e != 'other']
        self.emotion_probs = {e: 0.0 for e in COARSE_EMOTIONS + FINE_EMOTIONS}
        self.post_score    = 0.0
        self.post_stat     = None
        self.fidget_score  = {s: 0.0 for s in ["NORMAL","TAPPING","PINCHING","FIDGETING"]}

        # Detector instances
        self.posture = PostureDetector()
        self.fidget  = HandFidgetDetector()

        # Variables for the checkboxes
        self.var_emo        = tk.BooleanVar(value=False)
        self.var_vis_emo    = tk.BooleanVar(value=False)
        self.var_post       = tk.BooleanVar(value=False)
        self.var_vis_post   = tk.BooleanVar(value=False)
        self.var_fidget     = tk.BooleanVar(value=False)
        self.var_vis_fidget = tk.BooleanVar(value=False)

        self.update_counter = 0

        # Initialize the three charts
        self._init_emotion_chart()
        self._init_posture_chart()
        self._init_fidget_chart()

        # Build the GUI
        self.build_ui()

        # Start the video loop and worker thread
        self.video_after_id = self.root.after(10, self.video_loop)
        threading.Thread(target=self.worker_loop, daemon=True).start()

    # --- Chart initializers (compact size) ------------------------------------

    def _init_emotion_chart(self):
        self.fig_emo, self.ax_emo = plt.subplots(figsize=(4,3), dpi=80)
        labels = self.emo_labels
        colors = [EMO_BAR_COLORS[e] for e in labels]
        self.bars_emo = self.ax_emo.bar(labels, [0]*len(labels), color=colors)
        self.ax_emo.set_ylim(0,1)
        self.ax_emo.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        self.ax_emo.set_title("Emotions", fontsize=10)
        self.ax_emo.set_ylabel("Prob.", fontsize=9)
        self.fig_emo.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.30)

    def _init_posture_chart(self):
        self.fig_post, self.ax_post = plt.subplots(figsize=(4,3), dpi=80)
        labels = ["Good","Slight","Moderate","Severe"]
        self.bars_post = self.ax_post.bar(labels, [0]*4, color=POST_BAR_COLORS)
        self.ax_post.set_ylim(0,1)
        self.ax_post.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        self.ax_post.set_title("Posture", fontsize=10)
        self.ax_post.set_ylabel("Severity", fontsize=9)
        self.fig_post.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.30)

    def _init_fidget_chart(self):
        self.fig_fidg, self.ax_fidg = plt.subplots(figsize=(4,3), dpi=80)
        labels = ["NORMAL","TAPPING","PINCHING","FIDGETING"]
        self.bars_fidg = self.ax_fidg.bar(labels, [0]*4, color='lightgreen')
        self.ax_fidg.set_ylim(0,1)
        self.ax_fidg.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        self.ax_fidg.set_title("Fidgeting", fontsize=10)
        self.ax_fidg.set_ylabel("State", fontsize=9)
        self.fig_fidg.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.30)

    # --- Build the user interface (2×2 grid with bottom-right empty) ------------

    def build_ui(self):
        self.root.configure(bg="#2c2c2c")

        # Main split: left for video, right for charts
        main = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg="#2c2c2c")
        main.pack(fill=tk.BOTH, expand=True)

        # Left pane: live video
        video_frame = tk.Frame(main, bg="#1e1e1e")
        main.add(video_frame, stretch='always')
        self.canvas = tk.Canvas(video_frame, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right pane: container for 2×2 grid of charts
        charts_container = tk.Frame(main, bg="#2c2c2c")
        main.add(charts_container, width=600)

        # Configure 2 columns and 2 rows
        for i in range(2):
            charts_container.columnconfigure(i, weight=1)
            charts_container.rowconfigure(i, weight=1)

        # --- Emotions panel ---
        emo_panel = tk.LabelFrame(
            charts_container,
            text="Emotions",
            font=("Helvetica", 11, "bold"),
            padx=3, pady=3,
            bg="#3a3a3a", fg="white"
        )
        emo_panel.grid(row=0, column=0, sticky="nsew", padx=3, pady=3)

        # Checkboxes for emotions
        tk.Checkbutton(
            emo_panel, text="Enable",
            variable=self.var_emo,
            bg="#3a3a3a", fg="white", selectcolor="#2c2c2c"
        ).pack(anchor="nw", padx=5, pady=(5,0))

        tk.Checkbutton(
            emo_panel, text="Show Chart",
            variable=self.var_vis_emo,
            bg="#3a3a3a", fg="white", selectcolor="#2c2c2c"
        ).pack(anchor="nw", padx=5, pady=(0,5))

        # Embed the emotion chart
        self.canvas_emo = FigureCanvasTkAgg(self.fig_emo, master=emo_panel)
        self.canvas_emo.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Posture panel ---
        post_panel = tk.LabelFrame(
            charts_container,
            text="Posture",
            font=("Helvetica", 11, "bold"),
            padx=3, pady=3,
            bg="#3a3a3a", fg="white"
        )
        post_panel.grid(row=0, column=1, sticky="nsew", padx=3, pady=3)

        tk.Checkbutton(
            post_panel, text="Enable",
            variable=self.var_post,
            bg="#3a3a3a", fg="white", selectcolor="#2c2c2c"
        ).pack(anchor="nw", padx=5, pady=(5,0))

        tk.Checkbutton(
            post_panel, text="Show Chart",
            variable=self.var_vis_post,
            bg="#3a3a3a", fg="white", selectcolor="#2c2c2c"
        ).pack(anchor="nw", padx=5, pady=(0,5))

        self.canvas_post = FigureCanvasTkAgg(self.fig_post, master=post_panel)
        self.canvas_post.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Fidgeting panel ---
        fidg_panel = tk.LabelFrame(
            charts_container,
            text="Fidgeting",
            font=("Helvetica", 11, "bold"),
            padx=3, pady=3,
            bg="#3a3a3a", fg="white"
        )
        fidg_panel.grid(row=1, column=0, sticky="nsew", padx=3, pady=3)

        tk.Checkbutton(
            fidg_panel, text="Enable",
            variable=self.var_fidget,
            bg="#3a3a3a", fg="white", selectcolor="#2c2c2c"
        ).pack(anchor="nw", padx=5, pady=(5,0))

        tk.Checkbutton(
            fidg_panel, text="Show Chart",
            variable=self.var_vis_fidget,
            bg="#3a3a3a", fg="white", selectcolor="#2c2c2c"
        ).pack(anchor="nw", padx=5, pady=(0,5))

        self.canvas_fidg = FigureCanvasTkAgg(self.fig_fidg, master=fidg_panel)
        self.canvas_fidg.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bottom-right empty cell for spacing
        empty = tk.Frame(charts_container, bg="#2c2c2c")
        empty.grid(row=1, column=1, sticky="nsew", padx=3, pady=3)

    # --- Video capture and display loop -----------------------------------------

    def video_loop(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            # If frame not read, try reopening the camera
            self.cap.release()
            time.sleep(0.5)
            self.cap = cv2.VideoCapture(0)
            self.video_after_id = self.root.after(30, self.video_loop)
            return

        # Mirror the image
        frame = cv2.flip(frame, 1)
        self.last_frame = frame.copy()
        vis = frame.copy()

        # Overlay emotion boxes if requested
        if self.var_vis_emo.get():
            dominant = max(self.emotion_probs.items(), key=lambda x: x[1])[0]
            gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
            for (x, y, w, h) in face_cascade.detectMultiScale(gray,1.1,5,minSize=(30,30)):
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(vis, dominant, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Overlay posture if requested
        if self.var_vis_post.get():
            vis, _ = self.posture.process_frame(vis)

        # Overlay fidget gestures if requested
        if self.var_vis_fidget.get():
            vis = self.fidget.process_frame(vis)

        # Convert to PhotoImage and display
        img = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

        # Schedule next frame
        self.video_after_id = self.root.after(30, self.video_loop)

    # --- Background worker updating charts --------------------------------------

    def worker_loop(self):
        while self.running:
            frame = self.last_frame
            if frame is None:
                time.sleep(0.02)
                continue

            # --- Update emotion probabilities ---
            if self.var_emo.get():
                probs = detect_emotion_probs(frame)
                for k in self.emotion_probs:
                    self.emotion_probs[k] = probs.get(k, 0.0)
            else:
                for k in self.emotion_probs:
                    self.emotion_probs[k] = 0.0

            # --- Update posture score ---
            if self.var_post.get():
                _, keypoints = self.posture.process_frame(frame)
                status = self.posture.detect_slouching(keypoints).split()[0]
                self.post_stat  = status
                score_map = {"Good":0.0,"Slight":0.33,"Moderate":0.66,"Severe":1.0}
                self.post_score = score_map.get(status, 0.0)
            else:
                self.post_score = 0.0
                self.post_stat  = None

            # --- Update fidget state ---
            if self.var_fidget.get():
                self.fidget.process_frame(frame)
                state = "NORMAL"
                for hand in ["Left","Right"]:
                    hand_state = self.fidget.hands_data[hand]['state'].name
                    if hand_state == "FIDGETING":
                        state = "FIDGETING"
                        break
                    if hand_state == "PINCHING":
                        state = "PINCHING"
                    if hand_state == "TAPPING" and state == "NORMAL":
                        state = "TAPPING"
                for lbl in self.fidget_score:
                    self.fidget_score[lbl] = 1.0 if lbl == state else 0.0
            else:
                for lbl in self.fidget_score:
                    self.fidget_score[lbl] = 0.0

            # --- Refresh charts every 15 iterations ---
            self.update_counter += 1
            if self.update_counter % 15 == 0:
                # Emotion chart
                for bar, emo in zip(self.bars_emo, self.emo_labels):
                    bar.set_height(self.emotion_probs[emo])
                self.canvas_emo.draw()

                # Posture chart
                for bar, label in zip(self.bars_post, ["Good","Slight","Moderate","Severe"]):
                    height = self.post_score if label == self.post_stat else 0.0
                    bar.set_height(height)
                self.canvas_post.draw()

                # Fidget chart
                for bar, label in zip(self.bars_fidg, ["NORMAL","TAPPING","PINCHING","FIDGETING"]):
                    bar.set_height(self.fidget_score[label])
                self.canvas_fidg.draw()

            time.sleep(0.02)

    # --- Cleanup on close -------------------------------------------------------

    def on_close(self):
        if self.video_after_id:
            self.root.after_cancel(self.video_after_id)
        self.running = False
        self.cap.release()
        self.root.destroy()

# --- Entry point --------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app  = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
