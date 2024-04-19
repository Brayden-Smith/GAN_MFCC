
import matplotlib.pyplot as plt
import librosa
import numpy as np
import soundfile as sf

data = np.load("Achron, Joseph, Symphonic Variations and Sonata, Op.39, RtRrQwO2Hng.mid.wav.npy")

# Displaying the MFCCs:
plt.figure(figsize=(10, 4))

# Using librosa.display.specshow to show the MFCCs
librosa.display.specshow(data, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

audio = librosa.feature.inverse.mfcc_to_audio(data)
sf.write('stereo_file.wav', audio,44100, subtype='PCM_24')