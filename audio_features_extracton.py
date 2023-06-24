# audio_features_extracton.py: Extract features from wav files
# Written by Jingjing Nie, WSU

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import MidTermFeatures
from pyAudioAnalysis import audioSegmentation as aS
import parselmouth
from parselmouth.praat import call
import numpy as np

# Load an audio file
audio_path = 'path/data.wav'
[Fs, x] = audioBasicIO.read_audio_file(audio_path)
sound = parselmouth.Sound(audio_path)

# https://github.com/drfeinberg/PraatScripts/blob/master/Measure%20Pitch%2C%20HNR%2C%20Jitter%2C%20Shimmer%2C%20and%20Formants.ipynb
# Extract pitch-based features: pitch, jitter and shimmer
f0min = 75
f0max = 300
pitch = call(sound, "To Pitch", 0.0, f0min, f0max)
# Get jitter and shimmer values
pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
jitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
shimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
# Print the feature values
print("Pitch: ", pitch)
print("Jitter: ", jitter)
print("Shimmer: ", shimmer)


win_size = 5  # Window size (in seconds)
step_size = 0.5  # Step size (in seconds)
# get mid-term (segment) feature statistics 
# and respective short-term features:
mt, st, mt_n = MidTermFeatures.mid_feature_extraction(x, Fs, win_size * Fs, win_size * Fs, step_size * Fs, step_size * Fs)
print(f'signal duration {len(x)/Fs} seconds')

def mid_term_feature(feature, mt, mt_n ):
    # Get the index of the feature in the feature names list
    feature_index = mt_n.index(feature)
    # Get the feature values
    features = mt[feature_index]
    # Print the feature values
    print(feature)
    print(features)
# Example: Get Zero Crossing Rate, Energy, Chroma Deviation, and MFCCs
feature_name = "zcr_mean"
mid_term_feature(feature_name, mt, mt_n)
feature_name = "chroma_std_mean"
mid_term_feature(feature_name, mt, mt_n)
feature_name = "energy_mean"
mid_term_feature(feature_name, mt, mt_n)
feature_name = "mfcc_1_std"
mid_term_feature(feature_name, mt, mt_n)