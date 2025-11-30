import os
from transformers import pipeline

# Ensure transformers uses PyTorch and lowers verbosity
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

_emotion_classifier = None


def get_classifier():
    global _emotion_classifier
    if _emotion_classifier is None:
        print("Loading model...")
        _emotion_classifier = pipeline(
            "audio-classification",
            model="r-f/wav2vec-english-speech-emotion-recognition",
            device=-1  # CPU
        )
        print("✅ Model loaded!")
    return _emotion_classifier
