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

# ─── Găsește recursiv un fișier sub root ───────────────────────────────────────────
def find_file(root: str, filename: str) -> str:
    for dirpath, dirnames, filenames in os.walk(root):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    raise FileNotFoundError(f"Nu am găsit {filename} sub {root}")

# ─── Încarcă un modul Python dintr-un fișier ───────────────────────────────────────
def load_module_from_path(name: str, path: str):
    spec   = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# ─── Directorul de bază proiect ────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)

# ─── Localizăm și încărcăm modulele ───────────────────────────────────────────────
RD_PATH = find_file(ROOT, 'realtimedetection.py')
PP_PATH = find_file(ROOT, 'body_posture_detection.py')
FD_PATH = find_file(ROOT, 'main.py')

realtime_mod = load_module_from_path('emotions', RD_PATH)
posture_mod  = load_module_from_path('posture',  PP_PATH)
fidget_mod   = load_module_from_path('fidget',   FD_PATH)

# ─── Inițializăm modelele de emoții ────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EmotionCNN      = realtime_mod.EmotionCNN
COARSE_EMOTIONS = realtime_mod.COARSE_EMOTIONS
FINE_EMOTIONS   = realtime_mod.FINE_EMOTIONS
IMAGE_SIZE      = realtime_mod.IMAGE_SIZE

model_coarse = EmotionCNN(num_classes=len(COARSE_EMOTIONS)).to(device)
model_fine   = EmotionCNN(num_classes=len(FINE_EMOTIONS)).to(device)
model_coarse.load_state_dict(torch.load(os.path.join(ROOT, 'coarse_model.pth'),
                                       map_location=device))
model_fine.load_state_dict(torch.load(os.path.join(ROOT, 'fine_model.pth'),
                                      map_location=device))
model_coarse.eval()
model_fine.eval()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
emo_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def detect_emotion_probs(frame, threshold: float = 0.4):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30)
    )
    probs = {e:0.0 for e in COARSE_EMOTIONS + FINE_EMOTIONS}
    if len(faces) == 0:
        return probs

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
    face = Image.fromarray(face)
    tensor = emo_transform(face).unsqueeze(0).to(device)

    with torch.no_grad():
        out_c = model_coarse(tensor)
        pc   = F.softmax(out_c, dim=1)[0].cpu().numpy()
        for i, emo in enumerate(COARSE_EMOTIONS):
            probs[emo] = float(pc[i])
        if pc[COARSE_EMOTIONS.index('other')] > threshold:
            out_f = model_fine(tensor)
            pf    = F.softmax(out_f, dim=1)[0].cpu().numpy()
            for i, emo in enumerate(FINE_EMOTIONS):
                probs[emo] = float(pf[i])
    return probs

# ─── Setăm posture și fidgeting ───────────────────────────────────────────────────
PostureDetector    = posture_mod.PostureDetector
HandFidgetDetector = fidget_mod.HandFidgetDetector
HandState          = fidget_mod.HandState

# culori pentru fiecare emoție
EMO_BAR_COLORS = {
    "happy":    "#FFD700",
    "fear":     "#FF8C00",
    "angry":    "#FF4500",
    "neutral":  "#808080",
    "disgust":  "#800080",
    "sad":      "#1E90FF",
    "surprise": "#32CD32",
}
# culori pentru postura
POST_BAR_COLORS = ["#2E8B57", "#FFD700", "#FF8C00", "#FF4500"]  # Good→darkgreen,...Severe→orangered

class App:
    def __init__(self, root: tk.Tk):
        self.root    = root
        self.root.title("Live Monitor")
        self.root.attributes("-fullscreen", True)
        self.root.bind('<KeyPress-q>', lambda e: self.on_close())
        self.running = True

        self.cap        = cv2.VideoCapture(0)
        self.last_frame = None

        # etichete emoții fără 'other'
        self.emo_labels = [e for e in COARSE_EMOTIONS + FINE_EMOTIONS if e != 'other']

        # stări curente
        self.emotion_probs = {e:0.0 for e in COARSE_EMOTIONS + FINE_EMOTIONS}
        self.post_score    = 0.0
        self.post_stat     = None
        self.fidget_score  = {s:0.0 for s in ["NORMAL","TAPPING","PINCHING","FIDGETING"]}

        self.posture = PostureDetector()
        self.fidget  = HandFidgetDetector()

        self.var_emo        = tk.BooleanVar(value=False)
        self.var_vis_emo    = tk.BooleanVar(value=False)
        self.var_post       = tk.BooleanVar(value=False)
        self.var_vis_post   = tk.BooleanVar(value=False)
        self.var_fidget     = tk.BooleanVar(value=False)
        self.var_vis_fidget = tk.BooleanVar(value=False)

        self.build_ui()
        self.root.after(10, self.video_loop)
        threading.Thread(target=self.worker_loop, daemon=True).start()

    def build_ui(self):
        self.root.configure(bg="#2c2c2c")
        frm = tk.Frame(self.root, bg="#2c2c2c")
        frm.pack(fill=tk.BOTH, expand=True)

        # video (stânga)
        self.canvas = tk.Canvas(frm, width=640, height=480,
                                bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)

        # panou lateral (dreapta)
        side = tk.Frame(frm, bg="#2c2c2c")
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # controale
        ctrl = tk.LabelFrame(side, text="Controale",
                             font=("Helvetica",13,"bold"),
                             padx=10, pady=10,
                             bg="#3a3a3a", fg="white")
        ctrl.pack(fill=tk.X, pady=(0,10))
        for text, var in [
            ("Activează Emoții", self.var_emo),
            ("Vizualizare Emoții", self.var_vis_emo),
            ("Activează Postură", self.var_post),
            ("Vizualizare Postură", self.var_vis_post),
            ("Activează Fidgeting", self.var_fidget),
            ("Vizualizare Fidgeting", self.var_vis_fidget),
        ]:
            tk.Checkbutton(ctrl, text=text,
                           variable=var,
                           bg="#3a3a3a", fg="white",
                           selectcolor="#2c2c2c").pack(anchor="w")

        charts = tk.Frame(side, bg="#2c2c2c")
        charts.pack(fill=tk.BOTH, expand=True)

        # Emoții
        self.fig_emo, self.ax_emo = plt.subplots()
        self.fig_emo.set_size_inches(3,2)
        self.bars_emo = self.ax_emo.bar(
            self.emo_labels,
            [0]*len(self.emo_labels),
            color=[EMO_BAR_COLORS[e] for e in self.emo_labels]
        )
        self.ax_emo.set_ylim(0,1)
        self.ax_emo.set_xticks(range(len(self.emo_labels)))
        self.ax_emo.set_xticklabels(self.emo_labels,
                                    rotation=45, ha='right', fontsize=6)
        self.ax_emo.set_title("Emoții", fontsize=8)
        self.fig_emo.tight_layout()
        canvas_emo = FigureCanvasTkAgg(self.fig_emo, master=charts)
        canvas_emo.get_tk_widget().pack(fill=tk.X, pady=5)
        self.canvas_emo = canvas_emo

        # Fidgeting
        self.fig_fidg, self.ax_fidg = plt.subplots()
        self.fig_fidg.set_size_inches(3,2)
        labels_fidg = ["NORMAL","TAPPING","PINCHING","FIDGETING"]
        self.bars_fidg = self.ax_fidg.bar(labels_fidg, [0]*4, color='lightgreen')
        self.ax_fidg.set_ylim(0,1)
        self.ax_fidg.set_title("Fidgeting", fontsize=8)
        self.fig_fidg.tight_layout()
        canvas_fidg = FigureCanvasTkAgg(self.fig_fidg, master=charts)
        canvas_fidg.get_tk_widget().pack(fill=tk.X, pady=5)
        self.canvas_fidg = canvas_fidg

        # Postură
        self.fig_post, self.ax_post = plt.subplots()
        self.fig_post.set_size_inches(6,2)
        labels_post = ["Good","Slight","Moderate","Severe"]
        self.bars_post = self.ax_post.bar(labels_post, [0]*4, color=POST_BAR_COLORS)
        self.ax_post.set_ylim(0,1)
        self.ax_post.set_title("Postură", fontsize=8)
        self.fig_post.tight_layout()
        canvas_post = FigureCanvasTkAgg(self.fig_post, master=charts)
        canvas_post.get_tk_widget().pack(fill=tk.X, pady=5)
        self.canvas_post = canvas_post

    def video_loop(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            time.sleep(0.5)
            self.cap = cv2.VideoCapture(0)
            self.root.after(10, self.video_loop)
            return

        frame = cv2.flip(frame, 1)
        self.last_frame = frame.copy()

        vis = frame.copy()
        if self.var_vis_emo.get():
            emo = max(self.emotion_probs.items(), key=lambda x: x[1])[0]
            gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
            for (x,y,w,h) in face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30)):
                cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(vis, emo, (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        if self.var_vis_post.get():
            vis, _ = self.posture.process_frame(vis)
        if self.var_vis_fidget.get():
            vis = self.fidget.process_frame(vis)

        img = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        ph  = ImageTk.PhotoImage(img)
        self.canvas.create_image(0,0,anchor=tk.NW,image=ph)
        self.canvas.image = ph

        self.root.after(30, self.video_loop)

    def worker_loop(self):
        while self.running:
            frame = self.last_frame
            if frame is not None:
                # emoții
                if self.var_emo.get():
                    self.emotion_probs = detect_emotion_probs(frame)
                else:
                    for k in self.emotion_probs:
                        self.emotion_probs[k] = 0.0

                # postură
                if self.var_post.get():
                    _, kp = self.posture.process_frame(frame)
                    stat = self.posture.detect_slouching(kp).split()[0]
                    self.post_stat  = stat
                    score_map    = {"Good":0.0,"Slight":0.33,"Moderate":0.66,"Severe":1.0}
                    self.post_score = score_map.get(stat, 0.0)
                else:
                    self.post_score = 0.0
                    self.post_stat  = None

                # fidgeting
                if self.var_fidget.get():
                    self.fidget.process_frame(frame)
                    state = "NORMAL"
                    for hand in ["Left","Right"]:
                        st = self.fidget.hands_data[hand]['state'].name
                        if st=="FIDGETING":
                            state="FIDGETING"; break
                        if st=="PINCHING":
                            state="PINCHING"
                        if st=="TAPPING" and state=="NORMAL":
                            state="TAPPING"
                    for s in self.fidget_score:
                        self.fidget_score[s] = 1.0 if s==state else 0.0
                else:
                    for s in self.fidget_score:
                        self.fidget_score[s] = 0.0

                # update emoții
                for bar, emo in zip(self.bars_emo, self.emo_labels):
                    bar.set_height(self.emotion_probs[emo])
                self.canvas_emo.draw()

                # update fidgeting
                labels_fidg = ["NORMAL","TAPPING","PINCHING","FIDGETING"]
                for bar, lbl in zip(self.bars_fidg, labels_fidg):
                    bar.set_height(self.fidget_score[lbl])
                self.canvas_fidg.draw()

                # update postură
                labels_post = ["Good","Slight","Moderate","Severe"]
                for bar, lbl in zip(self.bars_post, labels_post):
                    bar.set_height(self.post_score if lbl==self.post_stat else 0.0)
                self.canvas_post.draw()

            time.sleep(0.2)

    def on_close(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

if __name__=="__main__":
    root = tk.Tk()
    app  = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
