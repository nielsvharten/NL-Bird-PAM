import os
from pathlib import Path
import time
import warnings
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import keras
from tqdm import tqdm
import csv
import sys
import librosa

sys.path.append(os.getcwd())

import config as cfg
path_dir = os.getcwd() + "\\BirdNET\\checkpoints\\V2.4\\BirdNET_GLOBAL_6K_V2.4_Model"
RANDOM = np.random.RandomState(42)


# implementation from BirdNET-Analyzer
def noise(sig, shape, amount=None):
    """Creates noise.

    Creates a noise vector with the given shape.

    Args:
        sig: The original audio signal.
        shape: Shape of the noise.
        amount: The noise intensity.

    Returns:
        An numpy array of noise with the given shape.
    """
    # Random noise intensity
    if amount == None:
        amount = RANDOM.uniform(0.1, 0.5)

    # Create Gaussian noise
    try:
        noise = RANDOM.normal(min(sig) * amount, max(sig) * amount, shape)
    except:
        noise = np.zeros(shape)

    return noise.astype("float32")


# implementation from BirdNET-Analyzer
def splitSignal(sig, rate, seconds, overlap, minlen):
    """Split signal with overlap.

    Args:
        sig: The original signal to be split.
        rate: The sampling rate.
        seconds: The duration of a segment.
        overlap: The overlapping seconds of segments.
        minlen: Minimum length of a split.

    Returns:
        A list of splits.
    """
    sig_splits = []

    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i : i + int(seconds * rate)]

        # End of signal?
        if len(split) < int(minlen * rate):
            break

        # Signal chunk too short?
        if len(split) < int(rate * seconds):
            split = np.hstack((split, noise(split, (int(rate * seconds) - len(split)), 0.5)))

        sig_splits.append(split)

    return sig_splits


def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    with open(class_map_csv_text) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        class_codes = [code[0] for code in csv_reader][1:]

    return class_codes


def frame_audio(audio_array: np.ndarray, window_size_s: float, hop_size_s: float, sample_rate: int) -> np.ndarray:
    """Helper function for framing audio for inference."""
    if window_size_s is None or window_size_s < 0:
        return audio_array[np.newaxis, :]

    frame_length = int(window_size_s * sample_rate)
    hop_length = int(hop_size_s * sample_rate)
    framed_audio = tf.signal.frame(audio_array, frame_length, hop_length, pad_end=True)

    return framed_audio

'''
def ensure_sample_rate(waveform, original_sample_rate, desired_sample_rate=32000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        waveform = tfio.audio.resample(waveform, original_sample_rate, desired_sample_rate)
  
    return desired_sample_rate, waveform
'''

def load(path: str):
    PERCH_SAMPLE_RATE = 32000
    wav_data = np.load(path)
    
    return frame_audio(wav_data, window_size_s=5.0, hop_size_s=1.0, sample_rate=PERCH_SAMPLE_RATE)


def divide_chunks(l, n): 
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 


class PerchWrapper():
    def __init__(self):
        perch_path = cfg.PERCH_MODEL_PATH
        labels_path = perch_path + "/assets/label.csv"

        self.classes = class_names_from_csv(labels_path)
        self.model = hub.load(perch_path)



    def embed_and_predict_waveform(self, dir: str, file: str):
        rec_path = "{}waveforms-32000/{}".format(dir, file)
        embeddings_path = "{}perch-embeddings/{}".format(dir, file)
        predictions_path = "{}perch-predictions/{}".format(dir, file)

        if os.path.exists(predictions_path) and os.path.exists(embeddings_path):
            return
        
        segments = load(rec_path)
        chunks = divide_chunks(segments, 20)

        embeddings = []
        predictions = []
        for chunk in chunks:
            logits, embedding = self.model.infer_tf(chunk)
            embeddings.extend(embedding.numpy())
            predictions.extend(tf.nn.softmax(logits).numpy())
        
        embeddings = np.array(embeddings)
        predictions = np.array(predictions)

        np.save(embeddings_path, embeddings)
        np.save(predictions_path, predictions)


    def embed_and_predict(self, dir):
        input_dir = dir + "waveforms-32000/"
        embeddings_dir = dir + "perch-embeddings/"
        predictions_dir = dir + "perch-predictions/"

        if not os.path.exists(input_dir): return
        waveform_files = next(os.walk(input_dir))[2]

        Path(embeddings_dir).mkdir(exist_ok=True)
        Path(predictions_dir).mkdir(exist_ok=True)

        for file in waveform_files:
            self.embed_and_predict_waveform(dir, file)


class BirdNETWrapper():
    def __init__(self):
        PBMODEL = keras.models.load_model(path_dir, compile=False)
        self.embedder = keras.Model(inputs=PBMODEL.model.inputs, outputs=PBMODEL.model.get_layer('GLOBAL_AVG_POOL').output)
        self.classifier = keras.Model(inputs=PBMODEL.model.get_layer('CLASS_DENSE_LAYER').input, outputs=PBMODEL.model.outputs)
 

    def embed_and_predict_waveform(self, dir: str, waveform_file: str):
        file_name = os.path.splitext(waveform_file)[0]
        waveform_path = "{}waveforms/{}.npy".format(dir, file_name)
        embeddings_path = "{}embeddings/{}.npy".format(dir, file_name)
        predictions_path = "{}predictions/{}.npy".format(dir, file_name)

        if os.path.exists(embeddings_path) and os.path.exists(predictions_path):
            return

        waveform = np.load(waveform_path)
        file_splits = splitSignal(waveform, 48000, 3, 2, 1)
        inputs = np.array(file_splits, dtype=np.float32)

        if len(inputs) > 0:
            embeddings = self.embedder.predict(inputs, verbose=0)
            predictions = self.classifier.predict(embeddings, verbose=0)

            np.save(embeddings_path, embeddings)
            np.save(predictions_path, predictions)
        else:
            print("Length of", file_name, "only", len(waveform) / 48000, "secs")


    def embed_and_predict(self, dir):
        input_dir = dir + "waveforms/"
        embeddings_dir = dir + "embeddings/"
        predictions_dir = dir + "predictions/"

        if not os.path.exists(input_dir): return

        Path(embeddings_dir).mkdir(exist_ok=True)
        Path(predictions_dir).mkdir(exist_ok=True)

        waveform_files = next(os.walk(input_dir))[2]

        for file in waveform_files:
            self.embed_and_predict_waveform(dir, file)