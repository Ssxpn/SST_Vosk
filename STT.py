import os
import sys
import json
import time
import pyaudio
from vosk import Model, KaldiRecognizer

# === Paramètres globaux ===
sample_rate = 16000
chunk_size = 8192
format = pyaudio.paInt16
channels = 1
SILENCE_TIMEOUT = 2  # secondes

# === Initialisation du modèle Vosk (à faire une seule fois) ===
def load_vosk_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "vosk-model-small-fr-pguyot-0.3")
    
    if not os.path.exists(model_path):
        print(f"❌ Modèle introuvable : {model_path}")
        sys.exit(1)

    print("✅ Modèle chargé.")
    return Model(model_path)

# === Vide le buffer audio ===
def clear_stream_buffer(stream, duration_ms=300):
    discard_frames = int((sample_rate / 1000) * duration_ms)
    try:
        stream.read(discard_frames, exception_on_overflow=False)
    except:
        pass

# === Lance une session STT ===
def run_speech_recognition(model):
    """Démarre une session unique de reconnaissance vocale, puis ferme le micro."""
    recognizer = KaldiRecognizer(model, sample_rate)

    # 🔌 Ouvre le micro dynamiquement
    p = pyaudio.PyAudio()
    stream = p.open(
        format=format,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size
    )

    stream.start_stream()
    clear_stream_buffer(stream)

    print("🎙️ Parle maintenant... (fin après silence)")
    last_voice_time = time.time()

    try:
        while True:
            data = stream.read(chunk_size, exception_on_overflow=False)

            if recognizer.AcceptWaveform(data):
                result_json = json.loads(recognizer.Result())
                text = result_json.get('text', '')
                if text.strip():
                    print("\r✅", text)
                    last_voice_time = time.time()
            else:
                partial_json = json.loads(recognizer.PartialResult())
                partial = partial_json.get('partial', '')
                if partial.strip():
                    sys.stdout.write(f"\r⏳ {partial}")
                    sys.stdout.flush()
                    last_voice_time = time.time()

            if time.time() - last_voice_time > SILENCE_TIMEOUT:
                print("\n🔇 Silence détecté. Fin de l'enregistrement.")
                break

    except KeyboardInterrupt:
        print("\n🛑 Arrêt manuel.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()