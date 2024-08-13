from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import os
import json
from sklearn.metrics import classification_report


def read_test1_lists(path):
    json_lists = []
    file_lists = os.listdir(path)
    for file_name in file_lists:
        if 'test1' in file_name:
            json_lists.append(os.path.join(path, file_name))
    return json_lists

def read_lists(path):
    json_lists = []
    file_lists = os.listdir(path)
    for file_name in file_lists:
            json_lists.append(os.path.join(path, file_name))
    return json_lists

def read_jsonl_file(file_path: str) -> list:
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def calculate_auc(true_labels, predicted_labels):
    """
    计算AUC（曲线下面积）
    
    参数:
    true_labels (list or array): 真实标签
    predicted_labels (list or array): 预测标签
    
    返回:
    float: AUC值
    """
    # 将标签转换为二进制形式
    lb = LabelBinarizer()
    true_labels_binary = lb.fit_transform(true_labels)
    predicted_labels_binary = lb.transform(predicted_labels)
    return roc_auc_score(true_labels_binary, predicted_labels_binary)

def calculate_multiclass_auc(true_labels, predicted_labels, num_classes=86):
    """
    计算多类分类问题的AUC（曲线下面积）
    
    参数:
    true_labels (list or array): 真实标签
    predicted_labels (list or array): 预测标签
    num_classes (int): 类别数
    
    返回:
    float: 平均AUC值
    """
    # 将标签转换为二进制矩阵
    lb = LabelBinarizer()
    lb.fit(range(num_classes))
    
    true_labels_binary = lb.transform(true_labels)
    predicted_labels_binary = lb.transform(predicted_labels)

    # 计算每个类别的AUC
    aucs = []
    for i in range(num_classes):
        if np.sum(true_labels_binary[:, i]) == 0 or np.sum(predicted_labels_binary[:, i]) == 0:
            # 如果某个类别没有正例或负例，跳过该类别
            continue
        auc = roc_auc_score(true_labels_binary[:, i], predicted_labels_binary[:, i])
        aucs.append(auc)
    
    # 计算平均AUC值
    if len(aucs) == 0:
        return 0
    return np.mean(aucs)

def calculate_metirc(y_true, y_pred):
    # 计算micro平均的precision, recall, f1-score
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    measure_result = classification_report(y_true, y_pred)
    print('measure_result = \n', measure_result)
    return precision, recall, f1, accuracy

def calculate_metrics(true_list, pred_list):
    # 确保真实列表和预测列表长度相同
    if len(true_list) != len(pred_list):
        raise ValueError("The length of true_list and pred_list must be the same.")
    
    # 将列表转换为二进制形式，True 代表 'Yes'，False 代表 'No'
    true_binary = [1 if x == 'Yes' else 0 for x in true_list]
    pred_binary = [1 if x == 'Yes' else 0 for x in pred_list]
    
    # 计算准确率
    accuracy = accuracy_score(true_binary, pred_binary)
    
    # 计算F1分数，需要指定pos_label为1，因为我们将'Yes'视为正类
    f1 = f1_score(true_binary, pred_binary, pos_label=1)
    
    # 计算AUC，需要预测概率，这里我们使用二元分类的预测结果来模拟
    # 假设pred_list已经是预测概率，我们取其阈值0.5来模拟
    # 如果pred_list是预测的类别，我们需要先将其转换为概率
    auc = roc_auc_score(true_binary, pred_binary)
    # auc = 1
    
    return accuracy, f1, auc

if __name__ == "__main__":
    base_path = 'save/ddi_dti'
    file_names = read_lists(base_path)
    for file_name in file_names:
        path = file_name + '/predict/generated_predictions.jsonl'
        data = read_jsonl_file(path)
        classify = ['Yes', 'No']
        label = []
        predict = []
        for item in data:
            if item['label'] in classify and item['predict'] in classify:
                label.append(item['label'])
                predict.append(item['predict'])
        accuracy, f1, auc = calculate_metrics(label, predict)
        print(f"{file_name.split('/')[-1].split('.')[0]}")
        print(f"Accuracy: {accuracy}")
        print(f"F1: {f1}")
        print(f"AUC: {auc}")