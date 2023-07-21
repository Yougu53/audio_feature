# audio_features_extracton.py: Extract features from wav files
# Written by Jingjing Nie, WSU
# Replace the folder path before run this code
# Output: csv file in the code folder
import os
import parselmouth
from parselmouth.praat import call
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import resampy
import csv
import io
import pandas as pd

model = hub.load('https://tfhub.dev/google/yamnet/1')

folder_path = 'path/to/your/folder' # replace the folder path
all_files = os.listdir(folder_path)

# Filter wave files with '.wav' extension
wave_files = [file for file in all_files if file.endswith('.wav')]

# https://github.com/drfeinberg/PraatScripts/blob/master/Measure%20Pitch%2C%20HNR%2C%20Jitter%2C%20and%20Shimmer.ipynb
# Extract pitch-based features: pitch, jitter, shimmer, and HNR
# This is the function to measure voice pitch
def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID) # read the sound
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    

    return meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer


# create lists to put the results
mean_F0_list = []
sd_F0_list = []
hnr_list = []
localJitter_list = []
localabsoluteJitter_list = []
rapJitter_list = []
ppq5Jitter_list = []
ddpJitter_list = []
localShimmer_list = []
localdbShimmer_list = []
apq3Shimmer_list = []
aqpq5Shimmer_list = []
apq11Shimmer_list = []
ddaShimmer_list = []
class_list = []
acr_list = []
flu_list = []

# Go through all the wave files in the folder and measure pitch
for wave_file in wave_files:
    sound = parselmouth.Sound(wave_file)
    audio, sample_rate = librosa.load(wave_file)
    # Resample the audio to the required sample rate of YAMNet (16 kHz)
    waveform = resampy.resample(audio, sample_rate, 16000)
    # Run the model, check the output.
    scores, embeddings, log_mel_spectrogram = model(waveform)
    scores.shape.assert_is_compatible_with([None, 521])
    embeddings.shape.assert_is_compatible_with([None, 1024])
    log_mel_spectrogram.shape.assert_is_compatible_with([None, 64])
    # Find the name of the class with the top score when mean-aggregated across frames.
    class_map_path = model.class_map_path().numpy()
    className = [display_name for (class_index, mid, display_name) in csv.reader(io.StringIO(tf.io.read_file(class_map_path).numpy().decode('utf-8')))]
    className = className[1:]  # Skip CSV header
    class_one = className[scores.numpy().mean(axis=0).argmax()]
    # Compute the Auto-correlation Function (ACR)
    acr = librosa.autocorrelate(audio)
    # Calculate the variance of the ACR
    acr_variance = acr.var()
    # Perform voice activity detection (VAD) to get unvoiced frames
    intervals = librosa.effects.split(audio, top_db=20)
    # Count the number of unvoiced frames and total frames
    num_unvoiced_frames = sum(interval[1] - interval[0] for interval in intervals)
    total_frames = len(audio)
    # Calculate the FLU feature
    flu_feature = num_unvoiced_frames / total_frames
    (meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer) = measurePitch(sound, 75, 500, "Hertz")
    class_list.append(class_one)
    mean_F0_list.append(meanF0) # make a mean F0 list
    sd_F0_list.append(stdevF0) # make a sd F0 list
    hnr_list.append(hnr)
    localJitter_list.append(localJitter)
    localabsoluteJitter_list.append(localabsoluteJitter)
    rapJitter_list.append(rapJitter)
    ppq5Jitter_list.append(ppq5Jitter)
    ddpJitter_list.append(ddpJitter)
    localShimmer_list.append(localShimmer)
    localdbShimmer_list.append(localdbShimmer)
    apq3Shimmer_list.append(apq3Shimmer)
    aqpq5Shimmer_list.append(aqpq5Shimmer)
    apq11Shimmer_list.append(apq11Shimmer)
    ddaShimmer_list.append(ddaShimmer)
    acr_list.append(acr_variance)
    flu_list.append(flu_feature)
df = pd.DataFrame(np.column_stack([class_list, mean_F0_list, sd_F0_list, hnr_list, localJitter_list, localabsoluteJitter_list, rapJitter_list, ppq5Jitter_list, ddpJitter_list, localShimmer_list, localdbShimmer_list, apq3Shimmer_list, aqpq5Shimmer_list, apq11Shimmer_list, ddaShimmer_list, acr_list, flu_list]),
                               columns=['class', 'meanF0Hz', 'stdevF0Hz', 'HNR', 'localJitter', 'localabsoluteJitter', 'rapJitter', 
                                        'ppq5Jitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 
                                        'apq11Shimmer', 'ddaShimmer', 'acr', 'flu'])  #add these lists to pandas in the right order
# Write out the updated dataframe
df.to_csv("processed_results.csv", index=False) # Change the csv name if the folder path is different
