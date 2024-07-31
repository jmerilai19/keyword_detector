import librosa
import os
import json
import tensorflow_datasets as tfds

DATASET_PATH = "data/mini_speech_commands/"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050 # 1 sec of audio with librosa's default sampling rate
DIRECTION_LABELS = ["down", "left", "right", "up"]

def prepare_dataset(dataset_path,
                    json_path,
                    n_mfcc=13, # coefficients
                    hop_length=512, # how many frames in sample window
                    n_fft=2048): # customary window for FFT
    # data dictionary
    data = {
        "mappings": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # list dirs that we want
    sub_dirs = []
    for label in DIRECTION_LABELS:
        sub_dirs.append(dataset_path + label)

    # loop through all sub-dirs
    i = 0
    for (dirpath, _, filenames) in os.walk(dataset_path):
        # only account direction directories
        if dirpath in sub_dirs:
            # add dir name to mappings
            category = dirpath.split("/")[-1]
            data["mappings"].append(category) 
            print(f"Processing {category}...")

            # loop through all the filenames and extract MFCCs
            for f in filenames:
                # load file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path)

                # ensure the audio file is at least 1 sec
                if len(signal) >= SAMPLES_TO_CONSIDER:
                    # enforce 1 sec long signal regardless of audio length
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # extract MFCCs
                    MFCCs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

                    # store data
                    data["labels"].append(i)
                    data["MFCCs"].append(MFCCs.T.tolist()) # cast numpy array to list for JSON serialization
                    data["files"].append(file_path)

            i += 1

    # store in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)
