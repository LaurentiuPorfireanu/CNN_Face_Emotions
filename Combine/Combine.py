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
from matplotlib.patches import Patch

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
# Colors for posture bar (Good, Slight, Moderate, Severe)
POST_BAR_COLORS = ["#2E8B57", "#FFD700", "#FF8C00", "#FF4500"]

# --- Main Application --------------------------------------------------------

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Live Monitor")

        # Full‐screen resolution (not fullscreen mode)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{sw}x{sh}+0+0")

        # Quit on 'q'
        self.root.bind('<KeyPress-q>', lambda e: self.on_close())
        self.running = True

        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.last_frame = None
        self.video_after_id = None

        # Data holders
        self.emo_labels    = [e for e in COARSE_EMOTIONS + FINE_EMOTIONS if e != 'other']
        self.emotion_probs = {e: 0.0 for e in COARSE_EMOTIONS + FINE_EMOTIONS}
        self.post_score    = 0.0
        self.post_stat     = None
        self.fidget_score  = {s: 0.0 for s in ["NORMAL","TAPPING","PINCHING","FIDGETING"]}

        # Detectors
        self.posture = PostureDetector()
        self.fidget  = HandFidgetDetector()

        # Checkbox vars
        self.var_emo        = tk.BooleanVar(value=False)
        self.var_vis_emo    = tk.BooleanVar(value=False)
        self.var_post       = tk.BooleanVar(value=False)
        self.var_vis_post   = tk.BooleanVar(value=False)
        self.var_fidget     = tk.BooleanVar(value=False)
        self.var_vis_fidget = tk.BooleanVar(value=False)

        self.update_counter = 0

        # Initialize charts
        self._init_emotion_chart()
        self._init_posture_chart()   # single‐bar stacked color chart
        self._init_fidget_chart()
        self._init_chart4()
        self._init_chart5()

        # Build UI and start loops
        self.build_ui()
        self.video_after_id = self.root.after(10, self.video_loop)
        threading.Thread(target=self.worker_loop, daemon=True).start()

    # --- Chart initializers ----------------------------------------------------

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
        """
        Initialize posture as a single bar whose color
        indicates the current state, plus a legend mapping colors.
        """
        self.fig_post, self.ax_post = plt.subplots(figsize=(4,3), dpi=80)

        # Single bar at x=0
        self.bar_post = self.ax_post.bar([0], [0], color=POST_BAR_COLORS[0])[0]
        self.ax_post.set_xlim(-0.5, 0.5)
        self.ax_post.set_ylim(0, 1)
        self.ax_post.set_xticks([])
        self.ax_post.set_title("Posture Severity", fontsize=10)
        self.ax_post.set_ylabel("Severity", fontsize=9)

        # Legend mapping colors → posture states
        labels = ["Good", "Slight", "Moderate", "Severe"]
        handles = [Patch(color=POST_BAR_COLORS[i], label=labels[i]) for i in range(4)]
        self.ax_post.legend(handles=handles, loc='upper right')

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

    def _init_chart4(self):
        self.fig4, self.ax4 = plt.subplots(figsize=(4,3), dpi=80)
        labels = []  # TODO
        self.bars4 = self.ax4.bar(labels, [], color='skyblue')
        self.ax4.set_ylim(0,1)
        self.ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        self.ax4.set_title("Chart 4", fontsize=10)
        self.ax4.set_ylabel("Value", fontsize=9)
        self.fig4.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.30)

    def _init_chart5(self):
        self.fig5, self.ax5 = plt.subplots(figsize=(4,3), dpi=80)
        labels = []  # TODO
        self.bars5 = self.ax5.bar(labels, [], color='salmon')
        self.ax5.set_ylim(0,1)
        self.ax5.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        self.ax5.set_title("Chart 5", fontsize=10)
        self.ax5.set_ylabel("Value", fontsize=9)
        self.fig5.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.30)

    # --- Build UI (2×3 grid) --------------------------------------------------

    def build_ui(self):
        self.root.configure(bg="#2c2c2c")
        main = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg="#2c2c2c")
        main.pack(fill=tk.BOTH, expand=True)

        # Video pane
        video_frame = tk.Frame(main, bg="#1e1e1e")
        main.add(video_frame, stretch='always')
        self.canvas = tk.Canvas(video_frame, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Charts pane
        charts = tk.Frame(main, bg="#2c2c2c")
        main.add(charts, width=900)
        for c in range(3): charts.columnconfigure(c, weight=1)
        for r in range(2): charts.rowconfigure(r, weight=1)

        # Emotions (0,0)
        emo = tk.LabelFrame(charts, text="Emotions",
                            font=("Helvetica",11,"bold"),
                            bg="#3a3a3a", fg="white", padx=3, pady=3)
        emo.grid(row=0, column=0, sticky="nsew", padx=3, pady=3)
        tk.Checkbutton(emo, text="Enable",    variable=self.var_emo,
                       bg="#3a3a3a", fg="white",
                       selectcolor="#2c2c2c").pack(anchor="nw",padx=5,pady=(5,0))
        tk.Checkbutton(emo, text="Show Chart",variable=self.var_vis_emo,
                       bg="#3a3a3a", fg="white",
                       selectcolor="#2c2c2c").pack(anchor="nw",padx=5,pady=(0,5))
        self.canvas_emo = FigureCanvasTkAgg(self.fig_emo, master=emo)
        self.canvas_emo.get_tk_widget().pack(fill=tk.BOTH,expand=True,padx=5,pady=5)

        # Posture (0,1)
        post = tk.LabelFrame(charts, text="Posture",
                             font=("Helvetica",11,"bold"),
                             bg="#3a3a3a", fg="white", padx=3, pady=3)
        post.grid(row=0, column=1, sticky="nsew", padx=3, pady=3)
        tk.Checkbutton(post, text="Enable",    variable=self.var_post,
                       bg="#3a3a3a", fg="white",
                       selectcolor="#2c2c2c").pack(anchor="nw",padx=5,pady=(5,0))
        tk.Checkbutton(post, text="Show Chart",variable=self.var_vis_post,
                       bg="#3a3a3a", fg="white",
                       selectcolor="#2c2c2c").pack(anchor="nw",padx=5,pady=(0,5))
        self.canvas_post = FigureCanvasTkAgg(self.fig_post, master=post)
        self.canvas_post.get_tk_widget().pack(fill=tk.BOTH,expand=True,padx=5,pady=5)

        # Fidgeting (1,0)
        fidg = tk.LabelFrame(charts, text="Fidgeting",
                             font=("Helvetica",11,"bold"),
                             bg="#3a3a3a", fg="white", padx=3, pady=3)
        fidg.grid(row=1, column=0, sticky="nsew", padx=3, pady=3)
        tk.Checkbutton(fidg, text="Enable",    variable=self.var_fidget,
                       bg="#3a3a3a", fg="white",
                       selectcolor="#2c2c2c").pack(anchor="nw",padx=5,pady=(5,0))
        tk.Checkbutton(fidg, text="Show Chart",variable=self.var_vis_fidget,
                       bg="#3a3a3a", fg="white",
                       selectcolor="#2c2c2c").pack(anchor="nw",padx=5,pady=(0,5))
        self.canvas_fidg = FigureCanvasTkAgg(self.fig_fidg, master=fidg)
        self.canvas_fidg.get_tk_widget().pack(fill=tk.BOTH,expand=True,padx=5,pady=5)

        # Chart4 (0,2)
        c4 = tk.LabelFrame(charts, text="Chart 4",
                           font=("Helvetica",11,"bold"),
                           bg="#3a3a3a", fg="white", padx=3, pady=3)
        c4.grid(row=0, column=2, sticky="nsew", padx=3, pady=3)
        self.canvas4 = FigureCanvasTkAgg(self.fig4, master=c4)
        self.canvas4.get_tk_widget().pack(fill=tk.BOTH,expand=True,padx=5,pady=5)

        # Chart5 (1,2)
        c5 = tk.LabelFrame(charts, text="Chart 5",
                           font=("Helvetica",11,"bold"),
                           bg="#3a3a3a", fg="white", padx=3, pady=3)
        c5.grid(row=1, column=2, sticky="nsew", padx=3, pady=3)
        self.canvas5 = FigureCanvasTkAgg(self.fig5, master=c5)
        self.canvas5.get_tk_widget().pack(fill=tk.BOTH,expand=True,padx=5,pady=5)

    # --- Video loop ------------------------------------------------------------

    def video_loop(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            time.sleep(0.5)
            self.cap = cv2.VideoCapture(0)
            self.video_after_id = self.root.after(30, self.video_loop)
            return

        frame = cv2.flip(frame, 1)
        self.last_frame = frame.copy()
        vis = frame.copy()

        if self.var_vis_emo.get():
            dom = max(self.emotion_probs.items(), key=lambda x: x[1])[0]
            gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
            for (x, y, w, h) in face_cascade.detectMultiScale(gray,1.1,5,minSize=(30,30)):
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(vis, dom, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if self.var_vis_post.get():
            vis, _ = self.posture.process_frame(vis)

        if self.var_vis_fidget.get():
            vis = self.fidget.process_frame(vis)

        img = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

        self.video_after_id = self.root.after(30, self.video_loop)

    # --- Worker loop -----------------------------------------------------------

    def worker_loop(self):
        while self.running:
            frame = self.last_frame
            if frame is None:
                time.sleep(0.02)
                continue

            # Emotions
            if self.var_emo.get():
                probs = detect_emotion_probs(frame)
                for k in self.emotion_probs:
                    self.emotion_probs[k] = probs.get(k, 0.0)
            else:
                for k in self.emotion_probs:
                    self.emotion_probs[k] = 0.0

            # Posture
            if self.var_post.get():
                _, kps = self.posture.process_frame(frame)
                status = self.posture.detect_slouching(kps).split()[0]
                self.post_stat  = status
                score_map = {"Good":0.0,"Slight":0.33,"Moderate":0.66,"Severe":1.0}
                self.post_score = score_map.get(status, 0.0)
            else:
                self.post_score = 0.0
                self.post_stat  = None

            # Fidget
            if self.var_fidget.get():
                self.fidget.process_frame(frame)
                state = "NORMAL"
                for hand in ["Left","Right"]:
                    hs = self.fidget.hands_data[hand]['state'].name
                    if hs == "FIDGETING":
                        state = "FIDGETING"; break
                    if hs == "PINCHING":
                        state = "PINCHING"
                    if hs == "TAPPING" and state == "NORMAL":
                        state = "TAPPING"
                for lbl in self.fidget_score:
                    self.fidget_score[lbl] = 1.0 if lbl == state else 0.0
            else:
                for lbl in self.fidget_score:
                    self.fidget_score[lbl] = 0.0

            # Update charts every 15 loops
            self.update_counter += 1
            if self.update_counter % 15 == 0:
                # Emotions
                for bar, emo in zip(self.bars_emo, self.emo_labels):
                    bar.set_height(self.emotion_probs[emo])
                self.canvas_emo.draw()

                # Posture single bar
                self.bar_post.set_height(self.post_score)
                colors = ["Good","Slight","Moderate","Severe"]
                if self.post_stat in colors:
                    idx = colors.index(self.post_stat)
                    self.bar_post.set_color(POST_BAR_COLORS[idx])
                else:
                    self.bar_post.set_color("gray")
                self.canvas_post.draw()

                # Fidget
                for bar, lbl in zip(self.bars_fidg, ["NORMAL","TAPPING","PINCHING","FIDGETING"]):
                    bar.set_height(self.fidget_score[lbl])
                self.canvas_fidg.draw()

                # Chart4 / Chart5 updates (if any)...
                # for bar, val in zip(self.bars4, your_vals4): bar.set_height(val)
                # self.canvas4.draw()
                # for bar, val in zip(self.bars5, your_vals5): bar.set_height(val)
                # self.canvas5.draw()

            time.sleep(0.02)

    # --- Cleanup ---------------------------------------------------------------

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
