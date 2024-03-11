# 每个标签对应多个特征样本，相当于提前输入一些病人的数据
# 读取原始文件的第一列并保存到新文件中
with open("LIWC_selected_features.csv", "r") as f:
    lines = f.readlines()

with open("new_label.csv", "w") as f:
    for line in lines:
        values = line.strip().split(",")
        if len(values) >= 1:
            first_column = values[0]  # 获取第一列的值
            f.write(f"{first_column}\n")
# 读取特征集文件
feature_data = {}
with open("LIWC_selected_features.csv", "r") as f:
    header = f.readline ().strip ().split (",")
    for line in f:
        values = line.strip ().split (",")
        filename = values[0]
        features = []

        for x in values[1:]:
            if x:
                features.append (float (x))
            else:
                features.append (0.0)  # 如果特征值为空字符串，将其设置为0.0

        feature_data[filename] = features

# 读取标签集文件
label_data = {}
with open("dataset.csv", "r") as f:
    for line in f:
        values = line.strip().split(",")
        if len(values) >= 2:
            filename = values[0]
            label = values[1]
            label_data[filename] = label

# 创建一个字典来存储匹配后的数据
matched_data = {}

for filename, features in feature_data.items():
    prefix = filename.split("-")[0]  # 提取文件名前缀
    label = label_data.get(prefix)    # 获取对应的标签

    if label is not None:
        if label not in matched_data:
            matched_data[label] = []

        matched_data[label].append((filename, features))

# 确保每个标签对应的数量与特征样本数量相等
min_samples = min(len(samples) for samples in matched_data.values())
for label in matched_data:
    matched_data[label] = matched_data[label][:min_samples]

# 创建一个新的标签集字典，用于保存匹配后的标签数据
new_label_data = {}

for label, samples in matched_data.items():
    for filename, _ in samples:
        new_label_data[filename] = label

# 将新的标签数据保存到文件中
with open("multi_label_dataset.csv", "w") as f:
    for filename, label in new_label_data.items():
        f.write(f"{filename},{label}\n")