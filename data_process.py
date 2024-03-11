# 预处理audio_features, 读取文件并归一化

from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def audio_preprocess_and_save(input_file, output_file):
    # 读取数据
    data = pd.read_csv(input_file)

    # 提取 File 列
    files_1 = data['File']
    files_2 = data['label']
    print(files_2)
    # 提取特征
    #X = data.drop(columns=['File'])  # 去除文件名列
    X = data.drop(columns=['File', 'label'])
    #X = data.drop(columns=['label'])

    # 实例化标准化器
    scaler = StandardScaler()

    # 对特征进行标准化
    X_scaled = scaler.fit_transform(X)

    # 将标准化后的特征和 File 列重新合并，将 File 列放在第一列
    X_scaled_with_files = pd.DataFrame(X_scaled, columns=X.columns)
    X_scaled_with_files.insert(0, 'File', files_1)
    X_scaled_with_files.insert(1,'label', files_2)
    # 保存处理后的数据为新的CSV文件
    X_scaled_with_files.to_csv(output_file, index=False)

def LIWC_extract_features(input_file, output_file, selected_features):
    """
    从输入的CSV文件中抽取选定的特征列，并将结果保存到新的CSV文件中。

    参数：
    input_file (str)：输入的CSV文件路径。
    output_file (str)：输出的CSV文件路径，用于保存抽取后的特征数据。
    selected_features (list)：要抽取的特征列的列表。
    """
    # 读取数据
    data = pd.read_csv(input_file)

    # 提取文件名列
    filenames = data["Filename"]

    # 从数据中抽取选定的特征列
    selected_data = data[selected_features]

    # 创建 MinMaxScaler 对象进行归一化
    scaler = MinMaxScaler ()

    # 将 LIWC 特征数据进行归一化
    normalized_features = scaler.fit_transform (selected_data)

    # 创建归一化后的 DataFrame
    normalized_data = pd.DataFrame (normalized_features, columns=selected_features)

    # 合并文件名和归一化后的特征数据
    extracted_and_normalized_data = pd.concat ([filenames, normalized_data], axis=1)

    # 保存抽取后的数据为新的CSV文件
    extracted_and_normalized_data.to_csv(output_file, index=False)



# 音频特征处理
input_file = "audio_features.csv"       # 替换为你的输入数据文件路径
output_file = "processed_audio_features.csv"  # 替换为你要保存的处理后数据文件路径
audio_preprocess_and_save(input_file, output_file)


'''LIWC_input_file = "LIWC_Results.csv"             # 替换为你的输入数据文件路径
LIWC_output_file = "LIWC_selected_features.csv"    # 替换为你要保存的抽取后数据文件路径

selected_features1 = ['we', 'shehe', 'they', 'family', 'Social', 'affiliation', 'friend']  # 要抽取的特征列
selected_features2 = ['risk', 'cause', 'reward']
selected_features3 = ['emotion', 'Affect', 'tone_pos', 'Tone', 'tone_neg', 'emo_pos', 'emo_neg', 'emo_anx', 'emo_anger', 'emo_sad']
selected_features4 = ['memory', 'focuspresent', 'certitude', 'quantity', 'number']
selected_features5 = ['tentat', 'discrep']
selected_features6 = ['space', 'power']

# 合并所有特征列表
all_selected_features = (
    selected_features1 +
    selected_features2 +
    selected_features3 +
    selected_features4 +
    selected_features5 +
    selected_features6
)
print(all_selected_features)

# 运行特征提取
LIWC_extract_features(LIWC_input_file, LIWC_output_file, all_selected_features)'''