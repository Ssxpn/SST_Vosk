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

def initialize_speech_recognition():
    """Initialise le modèle Vosk et le micro sans lancer l'écoute."""
    model_path = os.path.join(os.path.dirname(__file__), "models", "vosk-model-small-fr-pguyot-0.3")
    
    if not os.path.exists(model_path):
        print(f"❌ Modèle introuvable : {model_path}")
        sys.exit(1)

    print("✅ Modèle chargé.")

    model = Model(model_path)
    recognizer = KaldiRecognizer(model, sample_rate)

    p = pyaudio.PyAudio()
    stream = p.open(
        format=format,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size
    )

    return recognizer, stream, p

def clear_stream_buffer(stream, duration_ms=300):
    """Vide le buffer du micro en lisant et jetant les données."""
    frames_to_discard = int((sample_rate / 1000) * duration_ms)
    stream.read(frames_to_discard, exception_on_overflow=False)

def run_speech_recognition(recognizer, stream):
    """Lance la reconnaissance vocale jusqu'à détection de silence."""
    print("🎙️ Parle maintenant... (le programme s’arrêtera après silence)")
    clear_stream_buffer(stream)

    stream.start_stream()

    last_voice_time = time.time()

    try:
        while True:
            data = stream.read(chunk_size, exception_on_overflow=False)

            if recognizer.AcceptWaveform(data):
                result_json = json.loads(recognizer.Result())
                text = result_json.get('text', '')
                if text.strip():
                    print("\r" + text, end='\n')
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


def cleanup_micro(p, stream):
    """Arrête et ferme proprement le micro."""
    stream.stop_stream()
    stream.close()
    p.terminate()