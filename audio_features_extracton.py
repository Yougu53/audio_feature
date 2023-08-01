# audio_features_extracton.py: Extract features from wav files
# Written by Jingjing Nie, Washington State University
# Replace the folder path before run this code
# Output: csv file of features
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import MidTermFeatures
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
import speech_recognition as sr
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

# https://stackoverflow.com/questions/54945100/how-to-convert-speech-to-text-in-python-input-from-audio-file
def speech_to_text(wav_file):
    recognizer = sr.Recognizer()

    with sr.AudioFile(wav_file) as source:
        audio = recognizer.record(source)  # Record the audio from the WAV file

    try:
        text = recognizer.recognize_google(audio)  # Use Google Web Speech API for speech-to-text conversion
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Error {e}"

# create lists to put the results
texts = []
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
mfcc1_list = []
mfcc2_list = []
mfcc3_list = []
mfcc4_list = []
mfcc5_list = []
mfcc6_list = []
mfcc7_list = []
mfcc8_list = []
mfcc9_list = []
mfcc10_list = []
mfcc11_list = []
mfcc12_list = []
mfcc13_list = []

# Go through all the wave files in the folder and measure pitch
for wave_file in wave_files:
    sound = parselmouth.Sound(wave_file)
    audio, sample_rate = librosa.load(wave_file)
    [Fs, x] = audioBasicIO.read_audio_file(wave_file)

    win_size = 5  # Window size (in seconds)
    step_size = 0.5  # Step size (in seconds)
    mt, st, mt_n = MidTermFeatures.mid_feature_extraction(x, Fs, win_size * Fs, win_size * Fs, step_size * Fs, step_size * Fs)
    
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
    # speech to text
    text = speech_to_text(wave_file)
    texts.append(text)
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
    #average_mfccs = sum(mfcc_lists) / len(mfcc_lists)
    mfcc1_list.append(sum(mt[8]) / len(mt[8]))
    mfcc2_list.append(sum(mt[9]) / len(mt[9]))
    mfcc3_list.append(sum(mt[10]) / len(mt[10]))
    mfcc4_list.append(sum(mt[11]) / len(mt[11]))
    mfcc5_list.append(sum(mt[12]) / len(mt[12]))
    mfcc6_list.append(sum(mt[13]) / len(mt[13]))
    mfcc7_list.append(sum(mt[14]) / len(mt[14]))
    mfcc8_list.append(sum(mt[15]) / len(mt[15]))
    mfcc9_list.append(sum(mt[16]) / len(mt[16]))
    mfcc10_list.append(sum(mt[17]) / len(mt[17]))
    mfcc11_list.append(sum(mt[18]) / len(mt[18]))
    mfcc12_list.append(sum(mt[19]) / len(mt[19]))
    mfcc13_list.append(sum(mt[20]) / len(mt[20]))
df = pd.DataFrame(np.column_stack([class_list, texts, mean_F0_list, sd_F0_list, hnr_list, localJitter_list, localabsoluteJitter_list, rapJitter_list, ppq5Jitter_list, ddpJitter_list, localShimmer_list, localdbShimmer_list, apq3Shimmer_list, aqpq5Shimmer_list, apq11Shimmer_list, ddaShimmer_list, 
                                   acr_list, flu_list, mfcc1_list, mfcc2_list, mfcc3_list, mfcc4_list, mfcc5_list,mfcc6_list,mfcc7_list, mfcc8_list, mfcc9_list, mfcc10_list, mfcc11_list, mfcc12_list,mfcc13_list]),
                               columns=['class', 'Text', 'meanF0Hz', 'stdevF0Hz', 'HNR', 'localJitter', 'localabsoluteJitter', 'rapJitter', 
                                        'ppq5Jitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 
                                        'apq11Shimmer', 'ddaShimmer', 'acr', 'flu', 'mfcc1', 'mfcc2', 'mfcc3','mfcc4', 'mfcc5', 'mfcc6','mfcc7','mfcc8','mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13'])  #add these lists to pandas in the right order
# Write out the updated dataframe
df.to_csv("processed_results.csv", index=False)