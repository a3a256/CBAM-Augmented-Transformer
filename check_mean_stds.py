import torch
import torchaudio
import numpy as np

import os

means = []
stds = []

for i in os.listdir("processed_audio"):
    wave, sr = torchaudio.load(os.path.join("processed_audio", i))
    means += [wave.mean().item()]
    stds += [wave.std().item()]

print("Mean:", np.mean(means))
print("Std:", np.mean(stds))