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

#	─── Găsește recursiv un fișier sub root ───────────────────────────────────────────
def find_file(root: str, filename: str) -> str:
	for dirpath, dirnames, filenames in os.walk(root):
		if filename in filenames:
			return os.path.join(dirpath, filename)
	raise FileNotFoundError(f"Nu am găsit {filename} sub {root}")

#	─── Încarcă un modul Python dintr-un fișier ───────────────────────────────────────
def load_module_from_path(name: str, path: str):
	spec   = importlib.util.spec_from_file_location(name, path)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module

#	─── Directorul de bază proiect ────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../Conferinta/Combine
ROOT     = os.path.dirname(THIS_DIR)                    # .../Conferinta

#	─── Localizăm și încărcăm modulele ───────────────────────────────────────────────
RD_PATH = find_file(ROOT, 'realtimedetection.py')
PP_PATH = find_file(ROOT, 'body_posture_detection.py')
FD_PATH = find_file(ROOT, 'main.py')

realtime_mod = load_module_from_path('emotions', RD_PATH)
posture_mod  = load_module_from_path('posture',  PP_PATH)
fidget_mod   = load_module_from_path('fidget',   FD_PATH)

#	─── Inițializăm modelele de emoții ────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EmotionCNN      = realtime_mod.EmotionCNN
COARSE_EMOTIONS = realtime_mod.COARSE_EMOTIONS
FINE_EMOTIONS   = realtime_mod.FINE_EMOTIONS
IMAGE_SIZE      = realtime_mod.IMAGE_SIZE

model_coarse = EmotionCNN(num_classes=len(COARSE_EMOTIONS)).to(device)
model_fine   = EmotionCNN(num_classes=len(FINE_EMOTIONS)).to(device)
model_coarse.load_state_dict(torch.load(os.path.join(ROOT, 'coarse_model.pth'),
									   map_location=device))
model_fine  .load_state_dict(torch.load(os.path.join(ROOT, 'fine_model.pth'),
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

def detect_emotion(frame, threshold: float = 0.4) -> str:
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
	if not len(faces):
		return "none"
	x, y, w, h = faces[0]
	face = gray[y:y+h, x:x+w]
	face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
	face = Image.fromarray(face)
	tensor = emo_transform(face).unsqueeze(0).to(device)
	with torch.no_grad():
		out_c   = model_coarse(tensor)
		probs_c = F.softmax(out_c, dim=1)[0]
		p_other = probs_c[COARSE_EMOTIONS.index('other')]
		if p_other > threshold:
			out_f = model_fine(tensor)
			idx   = torch.argmax(out_f, dim=1).item()
			return FINE_EMOTIONS[idx]
		else:
			idx = torch.argmax(probs_c[:3]).item()
			return COARSE_EMOTIONS[idx]

def overlay_emotion(frame, emotion):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 4)
		cv2.putText(frame, emotion, (x, y-10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
	return frame

#	─── Setăm posture și fidgeting ───────────────────────────────────────────────────
PostureDetector    = posture_mod.PostureDetector
HandFidgetDetector = fidget_mod.HandFidgetDetector
HandState          = fidget_mod.HandState

#	─── Culori pentru beculețe ───────────────────────────────────────────────────────
EMO_COLORS = {
	"angry":    "red",
	"disgust":  "purple",
	"fear":     "orange",
	"happy":    "yellow",
	"neutral":  "white",
	"sad":      "blue",
	"surprise": "green"
}
POST_COLORS = {
	"Severe":   "red",
	"Moderate": "orange",
	"Slight":   "yellow",
	"Good":     "green"
}
FIDG_COLORS = {
	"NORMAL":    "green",
	"FIDGETING": "red",
	"PINCHING":  "orange",
	"TAPPING":   "purple"
}

#	─── Aplicația Tkinter ────────────────────────────────────────────────────────────
class App:
	def __init__(self, root: tk.Tk):
		self.root    = root
		self.root.title("Live Monitor")
		self.root.geometry("1000x600")
		self.running = True
		
		# cameră
		self.cap        = cv2.VideoCapture(0)
		self.last_frame = None
		
		# frame counter & current emotion
		self.frame_count     = 0
		self.current_emotion = "none"

		# detectoare
		self.posture = PostureDetector()
		self.fidget  = HandFidgetDetector()

		# variabile UI
		self.var_emo        = tk.BooleanVar()
		self.var_post       = tk.BooleanVar()
		self.var_fidget     = tk.BooleanVar()
		self.var_vis_emo    = tk.BooleanVar()
		self.var_vis_post   = tk.BooleanVar()
		self.var_vis_fidget = tk.BooleanVar()

		self.emo_list  = list(EMO_COLORS.keys())
		self.post_list = list(POST_COLORS.keys())
		self.fidg_list = list(FIDG_COLORS.keys())
		self.bulbs     = {}

		self.build_ui()
		self.root.after(10, self.video_loop)
		threading.Thread(target=self.worker_loop, daemon=True).start()
	def build_ui(self):
		self.root.configure(bg="#2c2c2c")  # Dark mode

		frm = tk.Frame(self.root, bg="#2c2c2c")
		frm.pack(fill=tk.BOTH, expand=True)

		# ▶️ Video (stânga)
		self.canvas = tk.Canvas(frm, width=640, height=480, bg="#1e1e1e", highlightthickness=0)
		self.canvas.pack(side=tk.LEFT, padx=10, pady=10)

		# ▶️ Controale + becuri (dreapta)
		side = tk.Frame(frm, bg="#2c2c2c")
		side.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

		# Facem doua coloane
		left_column = tk.Frame(side, bg="#2c2c2c")
		left_column.grid(row=0, column=0, padx=20, pady=10, sticky="n")

		right_column = tk.Frame(side, bg="#2c2c2c")
		right_column.grid(row=0, column=1, padx=20, pady=10, sticky="n")

		# ▶️ Coloana stanga - Emoții
		emo_frame = tk.LabelFrame(left_column, text="Emoții", font=("Helvetica", 13, "bold"), padx=10, pady=10, bg="#3a3a3a", fg="white")
		emo_frame.pack(pady=10, fill="x")
		tk.Checkbutton(emo_frame, text="Activează Emoții", variable=self.var_emo, bg="#3a3a3a", fg="white", selectcolor="#2c2c2c").pack(anchor="w", pady=5)
		tk.Checkbutton(emo_frame, text="Vizualizare Emoții", variable=self.var_vis_emo, bg="#3a3a3a", fg="white", selectcolor="#2c2c2c").pack(anchor="w", pady=5)
		for emo in self.emo_list:
			self._add_bulb(emo_frame, emo)

		# ▶️ Coloana dreapta - Postură
		post_frame = tk.LabelFrame(right_column, text="Postură", font=("Helvetica", 13, "bold"), padx=10, pady=10, bg="#3a3a3a", fg="white")
		post_frame.pack(pady=10, fill="x")
		tk.Checkbutton(post_frame, text="Activează Postură", variable=self.var_post, bg="#3a3a3a", fg="white", selectcolor="#2c2c2c").pack(anchor="w", pady=5)
		tk.Checkbutton(post_frame, text="Vizualizare Postură", variable=self.var_vis_post, bg="#3a3a3a", fg="white", selectcolor="#2c2c2c").pack(anchor="w", pady=5)
		for p in self.post_list:
			self._add_bulb(post_frame, p)

		# ▶️ Coloana dreapta - Fidgeting
		fidg_frame = tk.LabelFrame(right_column, text="Fidgeting", font=("Helvetica", 13, "bold"), padx=10, pady=10, bg="#3a3a3a", fg="white")
		fidg_frame.pack(pady=10, fill="x")
		tk.Checkbutton(fidg_frame, text="Activează Fidgeting", variable=self.var_fidget, bg="#3a3a3a", fg="white", selectcolor="#2c2c2c").pack(anchor="w", pady=5)
		tk.Checkbutton(fidg_frame, text="Vizualizare Fidgeting", variable=self.var_vis_fidget, bg="#3a3a3a", fg="white", selectcolor="#2c2c2c").pack(anchor="w", pady=5)
		for f in self.fidg_list:
			self._add_bulb(fidg_frame, f)


	def _add_bulb(self, parent, label):
		fr = tk.Frame(parent)
		fr.pack(anchor="w", pady=2)
		tk.Label(fr, text=label, width=10).pack(side=tk.LEFT)
		l = tk.Label(fr, text="●", font=("Arial",16), fg="grey")
		l.pack(side=tk.LEFT)
		self.bulbs[label] = l

	def set_bulb(self, label, color=None):
		col = color if color else "grey"
		if label in self.bulbs:
			self.bulbs[label].config(fg=col)

	def video_loop(self):
		if not self.running:
			return

		ret, frame = self.cap.read()
		if not ret:
			# Nu am putut citi frame-ul: eliberăm și reconectăm camera după o pauză
			self.cap.release()

			time.sleep(0.5)
			self.cap = cv2.VideoCapture(0)
			self.root.after(10, self.video_loop)
			return

		# Flip și păstrăm ultimul frame
		frame = cv2.flip(frame, 1)
		self.last_frame = frame.copy()

		# actualizăm emoția o dată la 15 cadre
		self.frame_count = (self.frame_count + 1) % 15
		if self.frame_count == 0 and self.var_emo.get():
			self.current_emotion = detect_emotion(frame)

		# Aplicăm suprapuneri vizuale
		vis = frame.copy()
		if self.var_vis_emo.get():
			vis = overlay_emotion(vis, self.current_emotion)
		if self.var_vis_post.get():
			vis, _ = self.posture.process_frame(vis)
		if self.var_vis_fidget.get():
			vis = self.fidget.process_frame(vis)

		# Convertim pentru Tkinter și afișăm
		img = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
		img = Image.fromarray(img)
		ph  = ImageTk.PhotoImage(img)
		self.canvas.create_image(0, 0, anchor=tk.NW, image=ph)
		self.canvas.image = ph

		# Programăm următoarea apelare
		self.root.after(10, self.video_loop)

	def worker_loop(self):
		while self.running:
			frm = self.last_frame

			# ▶ Emoții → beculețe (folosim current_emotion)
			if self.var_emo.get() and frm is not None:
				for e in self.emo_list:
					self.set_bulb(e, None)
				if self.current_emotion in EMO_COLORS:
					self.set_bulb(self.current_emotion, EMO_COLORS[self.current_emotion])
			else:
				for e in self.emo_list:
					self.set_bulb(e, None)

			# ▶ Postură → beculețe
			if self.var_post.get() and frm is not None:
				_, kp = self.posture.process_frame(frm)
				stat = self.posture.detect_slouching(kp)
				cat = None
				if stat.startswith("Severe"):
					cat = "Severe"
				elif stat.startswith("Moderate"):
					cat = "Moderate"
				elif stat.startswith("Slight"):
					cat = "Slight"
				elif stat.startswith("Good"):
					cat = "Good"
				for p in self.post_list:
					self.set_bulb(p, None)
				if cat in POST_COLORS:
					self.set_bulb(cat, POST_COLORS[cat])
			else:
				for p in self.post_list:
					self.set_bulb(p, None)

			# ▶ Fidgeting → beculețe
			if self.var_fidget.get() and frm is not None:
				self.fidget.process_frame(frm)
				state = "NORMAL"
				for hand in ["Left","Right"]:
					st = self.fidget.hands_data[hand]['state'].name
					if st == "FIDGETING":
						state = "FIDGETING"
						break
					if st == "PINCHING":
						state = "PINCHING"
					if st == "TAPPING" and state == "NORMAL":
						state = "TAPPING"
				for f in self.fidg_list:
					self.set_bulb(f, None)
				if state in FIDG_COLORS:
					self.set_bulb(state, FIDG_COLORS[state])
			else:
				for f in self.fidg_list:
					self.set_bulb(f, None)

			time.sleep(0.2)

	def on_close(self):
		self.running = False
		self.cap.release()
		self.root.destroy()

if __name__ == "__main__":
	root = tk.Tk()
	app = App(root)
	root.protocol("WM_DELETE_WINDOW", app.on_close)
	root.mainloop()
