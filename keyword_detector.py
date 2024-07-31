import librosa
import numpy as np
import tensorflow as tf

MODEL_PATH = "model.keras"
SAMPLES_TO_CONSIDER = 22050 # 1 sec with librosa's default sampling rate

class _KeywordDetector:
    model = None
    _mapping = [
        "down",
        "left",
        "right",
        "up"
    ]
    _instance = None

    def predict(self, file_path):
        # extract MFCCs
        MFCCs = self.preprocess(file_path) # (num of segments, num of coefficients)

        # convert 2d MFCCs array into 4d array
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis] # (num of samples, num of segments, num of coefficients, depth (=1))

        # make prediction
        predictions = self.model.predict(MFCCs) # [ [0.1, 0.4, 0.8, 0.3] ]
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]

        return predicted_keyword
    
    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):
        # load the audio file
        signal, sr = librosa.load(file_path)

        # ensure consistency in the audio file length
        if len(signal) >= SAMPLES_TO_CONSIDER:
            signal = signal[:SAMPLES_TO_CONSIDER]
        else:
            # pad the signal with zeros if it's shorter than SAMPLES_TO_CONSIDER
            signal = np.pad(signal, (0, max(0, SAMPLES_TO_CONSIDER - len(signal))), 'constant')

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        return MFCCs.T

def keyword_detector():
    # singleton
    if _KeywordDetector._instance is None:
        _KeywordDetector._instance = _KeywordDetector()
        _KeywordDetector.model = tf.keras.models.load_model(MODEL_PATH)
    return _KeywordDetector._instance

if __name__ == "__main__":
    detector = keyword_detector()

    down_test = detector.predict("data/mini_speech_commands/down/0a9f9af7_nohash_0.wav")
    left_test = detector.predict("data/mini_speech_commands/left/0b09edd3_nohash_0.wav")
    right_test = detector.predict("data/mini_speech_commands/right/0ab3b47d_nohash_0.wav")
    up_test = detector.predict("data/mini_speech_commands/up/0ab3b47d_nohash_0.wav")

    print(f"Predicted: {down_test} (expected: down)")
    print(f"Predicted: {left_test} (expected: left)")
    print(f"Predicted: {right_test} (expected: right)")
    print(f"Predicted: {up_test} (expected: up)")
