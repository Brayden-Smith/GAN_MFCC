import librosa
import os

import numpy as np


directory = 'wav files'
max_samples = int(22050 * 120)


for filename in os.listdir(directory):
    # bring in audio then delete it
    audio, sample_rate = librosa.load("wav files/" + filename)
    os.remove("wav files/" + filename)

    #if greater than 2 minutes cut to 2 minutes but if smaller pad with zeros
    if len(audio) > max_samples:
        audio = audio[:max_samples]  # Trim to max duration
    else:
        audio = np.pad(audio, (0, max_samples - len(audio)), 'constant')

    # create mfccs data then add to combined np file
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate)
    np.save("npFiles/" + filename, mfccs)
