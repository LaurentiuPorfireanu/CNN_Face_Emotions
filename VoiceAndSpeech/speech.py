import torch
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk
import threading
import time
import queue
import warnings
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoTokenizer, AutoModelForSequenceClassification

from Helpers.StatusLight import StatusLight

warnings.filterwarnings('ignore')


class SpeechRecognitionModel:
    def __init__(self):
        self.sample_rate = 16000
        self.speech_model = None
        self.processor = None
        self.tokenizer_sentiment = None
        self.model_sentiment = None
        self.is_loaded = False

    def load_model(self):
        try:
            # Load Romanian speech-to-text model
            self.processor = Wav2Vec2Processor.from_pretrained("anton-l/wav2vec2-large-xlsr-53-romanian")
            self.speech_model = Wav2Vec2ForCTC.from_pretrained("anton-l/wav2vec2-large-xlsr-53-romanian")

            # Load sentiment analysis model
            self.tokenizer_sentiment = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
            self.model_sentiment = AutoModelForSequenceClassification.from_pretrained(
                "cardiffnlp/twitter-roberta-base-sentiment")

            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def process_audio(self, audio):
        if not self.is_loaded:
            return self.get_default_output()

        try:
            # Speech-to-text processing
            inputs = self.processor(audio, return_tensors="pt", sampling_rate=self.sample_rate, padding=True)
            with torch.no_grad():
                logits = self.speech_model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0].strip()

            if not transcription:
                return self.get_default_output()

            # Sentiment analysis
            inputs = self.tokenizer_sentiment(transcription, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model_sentiment(**inputs)

            scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            sentiment_scores = scores.detach().cpu().numpy()[0]

            # Map labels to sentiments
            labels = ["negative", "neutral", "positive"]
            sentiment = labels[np.argmax(sentiment_scores)]

            return sentiment, transcription, sentiment_scores

        except Exception as e:
            print(f"Error in processing: {e}")
            return self.get_default_output()

    def get_default_output(self):
        return "neutral", "", np.array([0.33, 0.34, 0.33])


class SpeechRecognitionWidget(ttk.Frame):
    def __init__(self, parent, width=400, height=250, **kwargs):
        ttk.Frame.__init__(self, parent, **kwargs)

        self.parent = parent
        self.width = width
        self.height = height

        self.model = SpeechRecognitionModel()

        self.sample_rate = 16000
        self.chunk_duration = 2.0  # 2 seconds per chunk like in original speech.py
        self.chunk_samples = int(self.chunk_duration * self.sample_rate)

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
        # No column 1 since we're removing the sound bar
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)

        # Title frame
        title_frame = ttk.Frame(self)
        title_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

        # Single title/sentiment label that will change based on state
        self.title_label = ttk.Label(
            title_frame,
            text="Speech Recognition (Romanian)",
            font=("Arial", 12, "bold")
        )
        self.title_label.pack(pady=5)

        # Create text display area for transcription
        self.create_transcription_frame(self)

        # Sound bar has been removed

        # Control panel at bottom
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

    def create_transcription_frame(self, parent):
        """Create frame with text display for speech transcription"""
        transcription_frame = ttk.Frame(
            parent,
            width=self.width,  # Now uses full width since sound bar is removed
            height=int(self.height * 0.85)
        )
        transcription_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=2, pady=2)
        transcription_frame.grid_propagate(False)

        transcription_frame.grid_columnconfigure(0, weight=1)
        transcription_frame.grid_rowconfigure(0, weight=1)

        # Text widget for displaying transcription
        self.text_display = tk.Text(
            transcription_frame,
            wrap=tk.WORD,
            height=10,
            width=40,
            font=("Arial", 10),
            state="disabled"
        )
        self.text_display.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        # Scrollbar for text widget
        scrollbar = ttk.Scrollbar(transcription_frame, command=self.text_display.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.text_display['yscrollcommand'] = scrollbar.set

    def create_sound_bar_frame(self, parent):
        """Create an empty frame (sound bar removed)"""
        # This method is kept for compatibility but the sound bar is removed
        pass

    def toggle_model(self):
        """Toggle model loading/unloading"""
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

            # Clear display
            self.text_display.config(state="normal")
            self.text_display.delete(1.0, tk.END)
            self.text_display.config(state="disabled")
            # Reset title to original text
            self.title_label.config(text="Speech Recognition (Romanian)")

    def toggle_recording(self):
        """Toggle recording state"""
        if not self.model.is_loaded or self.is_loading:
            return

        if not self.is_recording:
            self.is_recording = True
            self.start_button.config(text="Stop")
            self.status_light.set_state("active")

            # Change title to show we're ready for sentiment
            self.title_label.config(text="😐 Neutral")

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

            # Restore original title
            self.title_label.config(text="Speech Recognition (Romanian)")

            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for processing audio data"""
        if status:
            print(f"Audio callback status: {status}")

        if self.is_recording:
            self.audio_buffer = indata[:, 0].copy()

            # Removed sound level processing since we removed the bar

            current_time = time.time()
            if current_time - self.last_update_time > 0.2:
                threading.Thread(
                    target=self.process_audio_thread,
                    args=(self.audio_buffer.copy(),)
                ).start()
                self.last_update_time = current_time

    def process_audio_thread(self, audio_data):
        """Process audio in a separate thread"""
        try:
            sentiment, transcription, sentiment_scores = self.model.process_audio(audio_data)
            if transcription:  # Only update if there's actual transcription
                self.result_queue.put(("speech_result", (sentiment, transcription, sentiment_scores)))
        except Exception as e:
            print(f"Error processing audio: {e}")

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

                elif msg_type == "audio_level":
                    # We've removed the sound bar, so we don't need to process this anymore
                    pass

                elif msg_type == "speech_result":
                    sentiment, transcription, sentiment_scores = msg_data
                    self.update_display(sentiment, transcription)

        except queue.Empty:
            pass

        self.after(50, self.check_results)

    def update_display(self, sentiment, transcription):
        """Update the text display with new transcription and sentiment"""
        # Format sentiment text based on the sentiment value
        if sentiment == "positive":
            sentiment_text = "😊 Positive"
        elif sentiment == "negative":
            sentiment_text = "😞 Negative"
        else:
            sentiment_text = "😐 Neutral"

        # Update title with sentiment
        if self.is_recording:
            self.title_label.config(text=sentiment_text)

        # Update transcription text
        self.text_display.config(state="normal")
        self.text_display.insert(tk.END, f"{transcription}\n\n")
        self.text_display.see(tk.END)
        self.text_display.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Speech Recognition")
    root.geometry("500x200")

    speech_recognition_widget = SpeechRecognitionWidget(root, width=500, height=200)
    speech_recognition_widget.pack(fill=tk.BOTH, expand=True)

    root.mainloop()