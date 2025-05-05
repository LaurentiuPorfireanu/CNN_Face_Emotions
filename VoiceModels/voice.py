import torch
import numpy as np
import sounddevice as sd
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
import threading
import time
import queue
import warnings

warnings.filterwarnings('ignore')


class VoiceEmotionDetectorModel:
    def __init__(self):
        self.sample_rate = 16000
        self.model = None
        self.feature_extractor = None
        self.is_loaded = False

    def load_model(self):
        try:
            model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

            self.model.to('cpu')
            self.model.eval()

            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def process_audio(self, audio):
        if not self.is_loaded:
            return self.get_default_output()

        try:
            inputs = self.feature_extractor(
                audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True
            )

            inputs = {k: v.to('cpu') if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits[0].detach().cpu().numpy()

            if len(logits) < 3:
                logits = np.pad(logits, (0, 3 - len(logits)), 'constant')

            arousal, dominance, valence = logits[:3]

            if arousal > 0:
                if valence > 0:
                    emotion = "excited/happy"
                else:
                    emotion = "angry/anxious"
            else:
                if valence > 0:
                    emotion = "calm/content"
                else:
                    emotion = "sad/bored"

            intensity = abs(arousal) + abs(valence)
            details = {
                'arousal': float(arousal),
                'valence': float(valence),
                'dominance': float(dominance)
            }

            return emotion, intensity, details

        except Exception as e:
            print(f"Error in processing: {e}")
            return self.get_default_output()

    def get_default_output(self):
        return "neutral", 0.0, {'arousal': 0.0, 'valence': 0.0, 'dominance': 0.0}


class StatusLight(ttk.Frame):
    def __init__(self, parent, size=15, **kwargs):
        ttk.Frame.__init__(self, parent, width=size, height=size, **kwargs)
        self.size = size

        self.canvas = tk.Canvas(self, width=size, height=size,
                                highlightthickness=0, bg="#333333")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.state = "off"
        self.update_color()

    def set_state(self, state):
        """Set light state: 'off', 'loading', 'ready', 'active'"""
        self.state = state
        self.update_color()

    def update_color(self):
        color_map = {
            "off": "#666666",
            "loading": "#ff3333",
            "ready": "#ffcc00",
            "active": "#33cc33"
        }
        color = color_map.get(self.state, "#666666")

        self.canvas.delete("all")
        self.canvas.create_oval(2, 2, self.size - 2, self.size - 2,
                                fill=color, outline="#444444", width=1)


class VoiceEmotionDetector(ttk.Frame):
    def __init__(self, parent, width=300, height=200, **kwargs):
        ttk.Frame.__init__(self, parent, **kwargs)

        self.parent = parent
        self.width = width
        self.height = height

        self.model = VoiceEmotionDetectorModel()

        self.sample_rate = 16000
        self.chunk_duration = 0.5
        self.chunk_samples = int(self.chunk_duration * self.sample_rate)

        self.current_emotion = "neutral"
        self.current_dimensions = {'arousal': 0.0, 'valence': 0.0, 'dominance': 0.0}

        self.is_recording = False
        self.is_loading = False
        self.audio_buffer = np.zeros(self.chunk_samples)
        self.last_update_time = time.time()

        self.result_queue = queue.Queue()

        self.create_ui()
        self.check_results()

    def create_ui(self):
        self.grid_propagate(False)
        self.config(width=self.width, height=self.height)

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)

        self.create_bar_graph_frame(self)
        self.create_sound_bar_frame(self)

        control_frame = ttk.Frame(self)
        control_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

        control_frame.grid_columnconfigure(0, weight=1)
        control_frame.grid_columnconfigure(1, weight=1)
        control_frame.grid_columnconfigure(2, weight=0)

        self.load_button = ttk.Button(
            control_frame,
            text="Load Model",
            command=self.toggle_model,
            width=12
        )
        self.load_button.grid(row=0, column=0, padx=2, pady=2, sticky="e")

        self.start_button = ttk.Button(
            control_frame,
            text="Start",
            command=self.toggle_recording,
            state="disabled",
            width=12
        )
        self.start_button.grid(row=0, column=1, padx=2, pady=2, sticky="w")

        self.status_light = StatusLight(control_frame, size=15)
        self.status_light.grid(row=0, column=2, padx=5, pady=2, sticky="e")
        self.status_light.set_state("off")

    def create_sound_bar_frame(self, parent):
        sound_frame = ttk.Frame(
            parent,
            width=int(self.width * 0.15),
            height=int(self.height * 0.85)
        )
        sound_frame.grid(row=1, column=1, sticky="nsew", padx=2, pady=2)
        sound_frame.grid_propagate(False)

        sound_frame.grid_columnconfigure(0, weight=1)
        sound_frame.grid_rowconfigure(0, weight=1)

        self.sound_bar = ttk.Progressbar(
            sound_frame,
            orient=tk.VERTICAL,
            length=int(self.height * 0.6),
            mode='determinate'
        )
        self.sound_bar.grid(row=0, column=0, padx=2, pady=2, sticky="ns")

    def create_bar_graph_frame(self, parent):
        graph_frame = ttk.Frame(parent, width=int(self.width * 0.85), height=int(self.height * 0.85))
        graph_frame.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        graph_frame.grid_propagate(False)

        is_dark_theme = self.is_dark_theme()

        bg_color = '#2b2b2b' if is_dark_theme else 'white'
        text_color = 'white' if is_dark_theme else 'black'
        grid_color = '#404040' if is_dark_theme else '#cccccc'

        self.fig = Figure(figsize=(int(self.width * 0.75) / 100, int(self.height * 0.8) / 100), dpi=100)
        self.fig.patch.set_facecolor(bg_color)

        self.fig.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.15)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Emotion Dimensions", color=text_color, pad=5, fontsize=12)
        self.ax.set_ylim(-1, 1)
        self.ax.grid(True, alpha=0.3, color=grid_color)
        self.ax.axhline(y=0, color=grid_color, linestyle='-', alpha=0.3)

        self.ax.set_facecolor(bg_color)
        self.ax.spines['bottom'].set_color(text_color)
        self.ax.spines['top'].set_color(text_color)
        self.ax.spines['right'].set_color(text_color)
        self.ax.spines['left'].set_color(text_color)

        self.ax.tick_params(axis='x', colors=text_color, labelsize=8)
        self.ax.tick_params(axis='y', colors=text_color, labelsize=8)

        dimensions = ['arousal', 'valence', 'dominance']
        values = [0, 0, 0]
        colors = ['#3498db', '#2ecc71', '#9b59b6']
        self.bars = self.ax.bar(dimensions, values, color=colors)

        self.canvas = FigureCanvasTkAgg(self.fig, graph_frame)
        self.canvas.draw()
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

    def toggle_model(self):
        if self.is_loading or self.is_recording:
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

    def toggle_recording(self):
        if not self.model.is_loaded or self.is_loading:
            return

        if not self.is_recording:
            self.is_recording = True
            self.start_button.config(text="Stop")
            self.status_light.set_state("active")

            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_samples,
                dtype='float32'
            )
            self.stream.start()
        else:
            self.is_recording = False
            self.start_button.config(text="Start")
            self.status_light.set_state("ready")

            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()

            self.current_emotion = "neutral"
            self.current_dimensions = {'arousal': 0.0, 'valence': 0.0, 'dominance': 0.0}
            self.sound_bar['value'] = 0
            self.clear_plot()

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio callback status: {status}")

        if self.is_recording:
            self.audio_buffer = indata[:, 0].copy()

            rms = np.sqrt(np.mean(np.square(self.audio_buffer))) * 300
            self.result_queue.put(("audio_level", rms))

            current_time = time.time()
            if current_time - self.last_update_time > 0.2:
                threading.Thread(
                    target=self.process_audio_thread,
                    args=(self.audio_buffer.copy(),)
                ).start()
                self.last_update_time = current_time

    def process_audio_thread(self, audio_data):
        try:
            emotion, intensity, details = self.model.process_audio(audio_data)
            self.result_queue.put(("emotion_result", (emotion, intensity, details)))
        except Exception as e:
            print(f"Error processing audio: {e}")

    def check_results(self):
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

                elif msg_type == "audio_level":
                    level = min(100, max(0, msg_data))
                    self.sound_bar['value'] = level

                elif msg_type == "emotion_result":
                    emotion, intensity, details = msg_data
                    self.update_emotion_display(emotion, intensity, details)

        except queue.Empty:
            pass

        self.after(50, self.check_results)

    def update_emotion_display(self, emotion, intensity, details):
        self.current_emotion = emotion
        self.current_dimensions = details
        self.update_bar_graph()

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

    def clear_plot(self):
        """Clear the plot and reset to neutral state"""
        is_dark = self.is_dark_theme()
        text_color = 'white' if is_dark else 'black'
        grid_color = '#404040' if is_dark else '#cccccc'
        bg_color = '#2b2b2b' if is_dark else 'white'

        self.ax.clear()

        dimensions = ['arousal', 'valence', 'dominance']
        values = [0, 0, 0]
        colors = ['#3498db', '#2ecc71', '#9b59b6']

        self.ax.bar(dimensions, values, color=colors)

        self.ax.set_title("Emotion Dimensions", color=text_color, pad=5, fontsize=12)
        self.ax.set_ylim(-1, 1)
        self.ax.grid(True, alpha=0.3, color=grid_color)
        self.ax.axhline(y=0, color=grid_color, linestyle='-', alpha=0.3)

        self.ax.set_facecolor(bg_color)
        self.ax.spines['bottom'].set_color(text_color)
        self.ax.spines['top'].set_color(text_color)
        self.ax.spines['right'].set_color(text_color)
        self.ax.spines['left'].set_color(text_color)

        self.ax.tick_params(axis='x', colors=text_color, labelsize=8)
        self.ax.tick_params(axis='y', colors=text_color, labelsize=8)

        self.fig.patch.set_facecolor(bg_color)
        self.canvas.draw()

    def update_bar_graph(self):
        dimensions = ['arousal', 'valence', 'dominance']
        values = [self.current_dimensions[dim] for dim in ('arousal', 'valence', 'dominance')]

        is_dark = self.is_dark_theme()
        text_color = 'white' if is_dark else 'black'
        grid_color = '#404040' if is_dark else '#cccccc'
        bg_color = '#2b2b2b' if is_dark else 'white'

        colors = ['#ff4444' if v < 0 else '#44ff44' for v in values]

        self.ax.clear()

        self.ax.bar(dimensions, values, color=colors)

        title_text = f"{self.current_emotion}"
        self.ax.set_title(title_text, color=text_color, pad=5, fontsize=14)
        self.ax.set_ylim(-1, 1)
        self.ax.grid(True, alpha=0.3, color=grid_color)
        self.ax.axhline(y=0, color=grid_color, linestyle='-', alpha=0.3)

        self.ax.set_facecolor(bg_color)
        self.ax.spines['bottom'].set_color(text_color)
        self.ax.spines['top'].set_color(text_color)
        self.ax.spines['right'].set_color(text_color)
        self.ax.spines['left'].set_color(text_color)

        self.ax.tick_params(axis='x', colors=text_color, labelsize=8)
        self.ax.tick_params(axis='y', colors=text_color, labelsize=8)

        for i, v in enumerate(values):
            self.ax.text(i, v + 0.05 if v >= 0 else v - 0.1,
                         f"{v:.2f}",
                         ha='center',
                         fontweight='bold',
                         color=text_color,
                         fontsize=8)

        self.fig.patch.set_facecolor(bg_color)
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Emotion Detector")
    root.geometry("300x200")

    voice_emotion_detector = VoiceEmotionDetector(root, width=300, height=200)
    voice_emotion_detector.pack(fill=tk.BOTH, expand=True)

    root.mainloop()