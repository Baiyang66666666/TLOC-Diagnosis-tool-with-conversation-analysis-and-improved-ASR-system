import re
from pydub import AudioSegment
import soundfile
import glob
from pathlib import Path
import os
import webrtcvad
import numpy as np
import librosa
import pandas as pd
import soundfile as sf

def word_count(text):
    words = text.split ()
    character_count = len(words)
    return character_count

def calculate_speech_duration(audio_file_path):
    try:
        # Read the audio file using soundfile
        audio_data, sample_rate = sf.read(audio_file_path)

        # Calculate the duration of speech in seconds (assuming VAD is applied)
        # You may need to adjust the VAD logic to determine speech regions
        # For now, let's assume the entire audio is speech (adjust as needed)
        speech_duration = len(audio_data) / sample_rate

        return speech_duration
    except Exception as e:
        print(f"Error processing {audio_file_path}: {str(e)}")
        return None


def calculate_pause_features(audio_file_path):
    # Loading Audio Files
    y, sr = librosa.load(audio_file_path)

    # Calculation of short-time energy
    energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)

    # Setting the threshold (number of samples in 30 ms)
    pause_threshold_samples = int(0.03 * sr / 512)  # 512是hop_length

    # Count pauses greater than 30 milliseconds
    pauses = []
    current_pause = 0

    for e in energy[0]:
        if e < 0.01:  # Set an appropriate threshold based on audio characteristics
            current_pause += 1
        else:
            if current_pause > pause_threshold_samples:
                pauses.append(current_pause)
            current_pause = 0

    # get features
    average_pause_length = np.mean(pauses) * 512 / sr  # Convert to seconds
    total_pause_time = np.sum(pauses) * 512 / sr  # Convert to seconds
    audio_length = len (y) / sr  # Audio length in seconds
    pause_frequency = total_pause_time / audio_length
    return pause_frequency, average_pause_length, total_pause_time

def count_hesitations(text):
    hesitations = ["hm", "um", "ERM"]
    count = 0
    for hesitation in hesitations:
        count += text.count (hesitation)
    return count


def count_repetitions(text):
    words = text.split ()
    count = 0
    for i in range (2, len(words)):
        if words[i] == words[i - 1] or words[i] == words[i - 2]:
            count += 1
    return count


def check_keywords_presence(text):
    # Replace with a list of keywords related to uncertainty
    uncertainty_keywords = ['MAYBE', 'PROBABLY', 'SORT OF', 'MAY', 'UNSURE', 'UNCERTAIN', 'POSSIBLY', 'MAY', 'LIKE',
                            'MIGHT', 'PROBABLE', 'POSSIBLE']
    for keyword in uncertainty_keywords:
        if keyword in text:
            return 1
    return 0


# Define a function to read text from a file
def read_label_text(file_name):
    base_name = Path(file_name).stem
    print (base_name, 'baseanme')
    label_file_path = Path(label_text_path) / f"{base_name}.txt"
    try:
        with open (label_file_path, "r") as file:
            label_text = file.read ().strip ()
        return label_text
    except FileNotFoundError:
        raise FileNotFoundError('label text not found')

# read the wav file
#file_path = "D:/qby/Epilepsy_prediction/pcm_files_withlabel"
file_path = "D:\qby\dissertation_project/allfiles/pcm_files - Copy"

# Use glob to get a list of audio file paths in the directory
file_list = glob.glob (os.path.join (file_path, "*.wav"))  # Change "*.wav" to match the desired audio file extension

# get the text label
global label_text_path
#label_text_path = "D:/qby/Epilepsy_prediction/text_files"
label_text_path = "D:\qby\dissertation_project/allfiles/text_files"

sample_rate = 16000
index = 0

data = []
for file_name in file_list:
    #print ('file_name', file_name)
    speech, rate = soundfile.read(file_name)

    # read the label text
    audio_text = read_label_text(file_name)

    #print (f"Input Speech: {file_name}")
    # display(Audio(speech, rate=rate))
    # librosa.display.waveshow(speech, sr=rate)
    index += 1

    # Extract features
    hesitations_number = count_hesitations (audio_text)
    repetitions_number = count_repetitions (audio_text)
    keywords_presence = check_keywords_presence (audio_text)
    pause_frequency, average_pause_length, total_pause_time = calculate_pause_features (file_name)
    wordcounts = word_count(audio_text)
    speech_duration = calculate_speech_duration(file_name)


    # Storing stored feature values as a dictionary
    feature_data = {
        'File': os.path.basename(file_name),
        'Hesitations': hesitations_number,
        'Repetitions': repetitions_number,
        'Uncertainty keywords': keywords_presence,
        'Pause Frequency': pause_frequency,
        'Average Pause Length': average_pause_length,
        'Total Pause Time': total_pause_time,
        'word count': wordcounts,
        'speech_duration': speech_duration

    }
    data.append(feature_data)


feature_df = pd.DataFrame(data)


# Load the CSV file containing audio features
audio_features_df = feature_df

# Load the CSV file containing labels without column names
labels_df = pd.read_csv('new_label.csv', header=None)

# Create a mapping of file names (without extensions) to labels
file_label_mapping = dict(zip(labels_df.iloc[:, 0].str.split('.').str[0], labels_df.iloc[:, 1]))

# Extract the first column (file names) from audio_features_df
audio_file_names = audio_features_df['File'].str.split('.').str[0]

# Initialize a list to store the extracted labels
extracted_labels = []

# Look up labels for each file name (without extension) and store in the list
for file_name in audio_file_names:
    label = file_label_mapping.get(file_name, 'Unknown')  # 'Unknown' if not found in mapping
    extracted_labels.append(label)

# Insert the extracted labels behind the second column in the original data frame, with the previous columns panned back.
audio_features_df.insert(1, column='label', value=extracted_labels)

# Save the updated DataFrame back to the original "audio_features.csv" file, overwriting its contents
audio_features_df.to_csv('audio_features.csv', index=False)




