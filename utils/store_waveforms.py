import librosa
import numpy as np
import os
from pathlib import Path


def load_audio(path: str, sample_rate):
    sig, rate = librosa.load(path, sr=sample_rate, mono=True, res_type="kaiser_fast")
    assert rate == sample_rate
    
    return sig


def store_waveforms(dir: str, sample_rate=48000):    
    input_dir = dir + "audio/"
    output_dir = dir + "waveforms-{}/".format(sample_rate)
    
    if not os.path.exists(input_dir): return
    Path(output_dir).mkdir(exist_ok=True)
    
    audio_files = next(os.walk(input_dir))[2]

    for file in audio_files:
        output_path = "{}{}.npy".format(output_dir, os.path.splitext(file)[0])
        if os.path.exists(output_path): continue

        sig = load_audio(input_dir + file, sample_rate)
        np.save(output_path, sig)
