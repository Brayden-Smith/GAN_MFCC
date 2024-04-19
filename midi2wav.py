from midi2audio import FluidSynth
import os
import shutil


directory = 'midis'
fs = FluidSynth(sound_font="GeneralUser GS 1.471/GeneralUser GS v1.471.sf2")


#go through each midi file we have
for filename in os.listdir(directory):
    #turn the midi into a wav and then turn wav into data for librosa
    fs.midi_to_audio('midis/' + filename, 'wav files/' + filename + '.wav')
    os.remove('midis/' + filename)

    total, used, free = shutil.disk_usage("/")

    if free // (2 ** 30) < 2:
        with open("C:/Users/brayd/PycharmProjects/wav2mfcc/main.py") as file:
            exec(file.read())