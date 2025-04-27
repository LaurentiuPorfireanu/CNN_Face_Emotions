import torch
import sounddevice as sd
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Parametri generali
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 2  # secunde pe batch

# Încarcă modelul românesc de speech-to-text
processor = Wav2Vec2Processor.from_pretrained("anton-l/wav2vec2-large-xlsr-53-romanian")
model_speech = Wav2Vec2ForCTC.from_pretrained("anton-l/wav2vec2-large-xlsr-53-romanian")

# Încarcă modelul public de analiză de sentiment (emoție)
tokenizer_sentiment = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model_sentiment = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model_sentiment,
    tokenizer=tokenizer_sentiment
)

# Flag pentru oprire
stop_recording = False

def recognize_stream_with_emotion():
    """Recunoaște voce live + analizează emoția"""
    global stop_recording
    print("🎤 Recunoaștere live pornită! Apasă Ctrl+C pentru oprire...\n")

    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio = np.squeeze(indata)

        # Speech-to-text
        inputs = processor(audio, return_tensors="pt", sampling_rate=SAMPLE_RATE, padding=True)
        with torch.no_grad():
            logits = model_speech(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0].strip()

        if transcription:
            # Analiza de emoție
            emotion = sentiment_pipeline(transcription)[0]
            label = emotion['label']

            # Interpretăm label-ul
            if label == 'LABEL_0':
                emotion_text = "😞 Emoție: Negativă"
            elif label == 'LABEL_1':
                emotion_text = "😐 Emoție: Neutră"
            elif label == 'LABEL_2':
                emotion_text = "😊 Emoție: Pozitivă"
            else:
                emotion_text = f"🤔 Emoție necunoscută ({label})"

            # Afișăm frumos emoția și textul
            print(f"\n{emotion_text}\n\"{transcription}\"\n", flush=True)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32',
                            blocksize=int(SAMPLE_RATE * BLOCK_DURATION),
                            callback=callback):
            while not stop_recording:
                sd.sleep(100)
    except KeyboardInterrupt:
        stop_recording = True
        print("\n🛑 Recunoaștere oprită.")

if __name__ == "__main__":
    recognize_stream_with_emotion()
