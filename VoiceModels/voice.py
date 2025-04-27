import torch
import sounddevice as sd
import numpy as np
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
from termcolor import colored

# Înregistrare audio
def record_audio(duration=2, sample_rate=16000):
    print("🎙️ Vorbește...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return recording.squeeze()

# Bip sonor
def beep():
    print('\a')  # ASCII 7 beep

# Încărcăm modelul și extractorul
model_name = "superb/hubert-large-superb-er"
model = HubertForSequenceClassification.from_pretrained(model_name)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model.eval()

# Emoții suportate
emotions = ["angry", "happy", "sad", "neutral"]

# Culori pentru emoții
emotion_colors = {
    "angry": "red",
    "happy": "yellow",
    "sad": "blue",
    "neutral": "cyan"
}

duration = 2
sample_rate = 16000
last_emotion = None

print("🔁 Ascultare continuă pornită (Ctrl+C pentru oprire):\n")

try:
    while True:
        audio = record_audio(duration, sample_rate)

        inputs = feature_extractor(
            audio, sampling_rate=sample_rate, return_tensors="pt", padding=True
        )

        with torch.no_grad():
            logits = model(**inputs).logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            top_prob, predicted_id = torch.max(probabilities, dim=-1)

        predicted_emotion = emotions[predicted_id.item()]
        confidence = top_prob.item() * 100  # în procente

        color = emotion_colors.get(predicted_emotion, "white")
        emotion_text = colored(f"{predicted_emotion.upper()}", color)

        print(f"🧠 Emoție detectată: {emotion_text} cu încredere de {confidence:.1f}%\n")

        # Dacă s-a schimbat emoția -> beep
        if predicted_emotion != last_emotion and last_emotion is not None:
            print("\a🔔 Emoția s-a schimbat!")
        last_emotion = predicted_emotion

except KeyboardInterrupt:
    print("\n🛑 Ascultare oprită.")
