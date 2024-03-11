# 这个代码现在可以实现将病人的语音进行转录并计算WER,准确率高，可以有两种模式，分段时有没有overlap，有overlap的明显性能更好

import torch
import string
import os
import whisper
import warnings
import soundfile
import glob
from pathlib import Path


warnings.filterwarnings ("ignore")

model = whisper.load_model ("large")
fs = 16000  # @param {type:"integer"}
# check for GPU availability
device = 'cuda' if torch.cuda.is_available() else "cpu"

'''
# For the audio files less than 30 secs
def inference(audio_path, fs):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    return result.text
'''


#audio longer than 30 secs
def process_audio_segment(audio_segment, model):
    mel = whisper.log_mel_spectrogram(audio_segment).to(model.device)
    _, probs = model.detect_language(mel)
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    return result.text

def inference(audio_path, fs):
    audio = whisper.load_audio(audio_path)
    audio_duration = len(audio) / fs

    if audio_duration < 30:
        audio = whisper.pad_or_trim(audio)
        return process_audio_segment(audio, model)
    else:
        max_duration = 30  #Maximum segment dutation in seconds
        segment_results = []
        for start_time in range(0, len(audio), int(max_duration * fs)):
            end_time = start_time + int (max_duration * fs)
            audio_segment = audio[start_time:end_time]
            audio_segment = whisper.pad_or_trim(audio_segment)
            segment_text = process_audio_segment(audio_segment, model)
            segment_results.append (segment_text)
        return ' '.join (segment_results)

'''
def process_audio_segment(audio_segment, model):
    mel = whisper.log_mel_spectrogram(audio_segment).to(model.device)
    _, probs = model.detect_language(mel)
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    return result.text

def inference(audio_path, fs):
    audio = whisper.load_audio(audio_path)
    audio_duration = len(audio) / fs
    i = 0
    if audio_duration < 30:
        audio = whisper.pad_or_trim(audio)
        return process_audio_segment(audio,model)
    else:
        max_duration = 30  #Maximum segment dutation in seconds
        segment_results = []
        for start_time in range (0, len (audio), int (max_duration / 2 * fs)):
            end_time = start_time + int (max_duration * fs)
            audio_segment = audio[start_time:end_time]
            audio_segment = whisper.pad_or_trim (audio_segment)

            segment_text = process_audio_segment (audio_segment, model)
            segment_results.append (segment_text)
            print('start time', start_time)
            print(segment_results)

        final_results = []

        for i, result in enumerate (segment_results):
            if i == 0:
                final_results.append (result)
            else:
                prev_result = final_results[-1]
                common_part = ""
                common_part_found = False

                for j in range (min (len (prev_result), len (result))):
                    if prev_result[j] == result[j]:
                        common_part += prev_result[j]
                    else:
                        break

                # 判断common_part是否已经在之前的结果中出现过
                if common_part in ''.join (final_results):
                    last_part = result.replace (common_part, "", 1)
                else:
                    last_part = result

                final_results.append (last_part)

        final_text = ' '.join (final_results)
        return final_text
'''

def text_normalizer(text):
    text = text.upper()
    return text.translate(str.maketrans('', '', string.punctuation))


# Define a function to read text from a file
def read_label_text(file_name):
    base_name = Path (file_name).stem
    print (base_name, 'baseanme')
    label_file_path = Path(label_text_path) / f"{base_name}.txt"
    try:
        with open (label_file_path, "r") as file:
            label_text = file.read ().strip ()
        return label_text
    except FileNotFoundError:
        raise FileNotFoundError ('label text not found')


# Define a function to calculate WER
# Define a function to calculate WER
def calculate_wer(predicted_text, label_text):
    """
        Calculate the Word Error Rate (WER) between predicted and label texts.

        Args:
            predicted_text (str): The predicted text from ASR.
            label_text (str): The ground truth label text.

        Returns:
            float: The Word Error Rate (WER) as a percentage.
    """
    # convert the input texts into lists of words
    predicted_words = predicted_text.split ()
    label_words = label_text.split ()

    # Create a mapping of words to their frequencies in the reference text
    ref_word_freq = {}
    for word in label_words:
        ref_word_freq[word] = ref_word_freq.get (word, 0) + 1

    # Calculate the number of correct words
    correct_words = 0
    for word in predicted_words:
        if word in ref_word_freq and ref_word_freq[word] > 0:
            ref_word_freq[word] -= 1
            correct_words += 1
    # Calculate the word error rate
    # wer = 100 * total_word_errors/len(label_words)
    if len(label_words)==0:
        wer = 0
        return wer
    else:
        wer = 100 * (len (label_words) - correct_words) / len (label_words)
        return wer

def calculate_ril(input_text, output_text):
    """
    计算Relative Information Lost (RIL)

    Args:
        input_text (str): 输入文本。
        output_text (str): ASR系统输出的文本。

    Returns:
        float: Relative Information Lost (RIL)作为百分比表示的信息损失指标。
    """
    # 计算输入文本和输出文本的字符长度
    input_length = len(input_text)
    output_length = len(output_text)

    # 计算RIL
    if input_length==0:
        ril = 0
        return ril
    else:
        ril = ((input_length - output_length) / input_length) * 100.0
        return ril

    return ril

def calculate_wil(predicted_text, label_text):
    """
    计算Word Information Lost (WIL)

    Args:
        predicted_text (str): ASR系统预测的文本。
        label_text (str): 标签文本（参考文本）。

    Returns:
        float: Word Information Lost (WIL)作为百分比表示的信息损失指标。
    """
    # 将预测文本和标签文本转换为单词列表
    predicted_words = predicted_text.split()
    label_words = label_text.split()

    # 计算插入的额外单词数
    extra_words = len(predicted_words) - len(label_words)

    # 计算WIL
    if len(label_words)==0:
        wer = 0
        return wer
    else:
        wil = (extra_words / len(label_words)) * 100.0
        return wil


def calculate_wip(predicted_text, label_text):
    # 计算WIP（Word Information Preservation）
    predicted_words = set (text_normalizer (predicted_text).split ())
    label_words = set (text_normalizer (label_text).split ())

    preserved_words = predicted_words.intersection (label_words)
    # 计算WIL
    if len(label_words)==0:
        wer = 0
        return wer
    else:
        wip = (len (preserved_words) / len (label_words))*100
        return wip


# save the new text files generated
def save_inference_result(file_name, text):
    # 获取文件名（不包括路径和扩展名）
    base_name = Path (file_name).stem
    # 指定要保存推断文本的文件夹路径
    output_dir = "D:/qby/dissertation_project/large_infer"
    # 构建输出文件的完整路径
    output_file_path = os.path.join (output_dir, f"{base_name}.txt")

    # 打开文件并写入推断文本
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(text)

print ('*********** start ****************')
#输入要识别的音频文件路径
file_path = "D:/qby/dissertation_project/allfiles/pcm_files_quan"
print (file_path)
# Use glob to get a list of audio file paths in the directory
file_list = glob.glob (os.path.join (file_path, "*.wav"))  # Change "*.wav" to match the desired audio file extension

# get the text label
global label_text_path
label_text_path = "D:/qby/dissertation_project/allfiles/text_files"


wer_list = []
ril_list = []
wil_list = []
wip_list = []
index = 0
for file_name in file_list:
    print ('file_name', file_name)
    speech, rate = soundfile.read(file_name)
    assert rate == fs, "mismatch in sampling rate"
    #nbests = speech2text (speech)
    #text, *_ = nbests[0]
    text = inference(file_name, fs)
    # save the new text
    save_inference_result (file_name, text)
    # read the label text
    label_text = read_label_text(file_name)

    print (f"Input Speech: {file_name}")
    # display(Audio(speech, rate=rate))
    # librosa.display.waveshow(speech, sr=rate)
    index += 1
    # plt.savefig (f"wavform_epilepsyplot{index}.png")
    print (f"ASR hypothesis: {text_normalizer (text)}")
    print (f"label text: {text_normalizer (label_text)}")

    # Calculate the WER
    wer = calculate_wer(text_normalizer (text), text_normalizer (label_text))
    print (f"WER: {wer:.2f}%")
    ril = calculate_ril(text_normalizer (text), text_normalizer (label_text))
    wil = calculate_wil(text_normalizer (text), text_normalizer (label_text))
    wip = calculate_wip(text_normalizer (text), text_normalizer (label_text))

    wer_list.append (wer)
    ril_list.append(ril)
    wip_list.append(wip)
    wil_list.append(wil)
    print ("*" * 50)

# 计算整个数据集的平均WER
average_wer = sum(wer_list) / len(wer_list)
average_ril = sum(ril_list) / len(ril_list)
average_wip = sum(wip_list) / len(wip_list)
average_wil = sum(wil_list) / len(wil_list)
print(f"Average WER: {average_wer:.2f}%")
print(f"Average RIL: {average_ril:.2f}%")
print(f"Average WIP: {average_wip:.2f}%")
print(f"Average WIL: {average_wil:.2f}%")
