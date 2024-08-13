import json
import os
import csv
# 读取文件下的json文件名
def read_json_lists(path):
    json_lists = []
    file_lists = os.listdir(path)
    for file_name in file_lists:
            json_lists.append(os.path.join(path, file_name))
    return json_lists

def calculate_r2(y_true, y_pred):
    """
    计算R^2指标
    :param y_true: 真实值列表
    :param y_pred: 预测值列表
    :return: R^2值
    """
    ss_res = sum((y - y_pred[i]) ** 2 for i, y in enumerate(y_true))
    ss_tot = sum((y - sum(y_true) / len(y_true)) ** 2 for y in y_true)
    return 1 - (ss_res / ss_tot)


def calculate_mae(labels: list, predictions: list) -> float:
    """
    计算MAE指标
    :param labels: 真实值列表
    :param predictions: 预测值列表
    :return: MAE值
    """
    return sum(abs(label - pred) for label, pred in zip(labels, predictions)) / len(labels)

def read_jsonl_file(file_path: str) -> list:
    """
    读取JSON Lines文件
    :param file_path: 文件路径
    :return: 包含所有JSON对象的列表
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def main():
    # 假设JSON Lines文件路径为'data.jsonl'
    # file_path = '/zengdaojian/litianle/Mycode/Mol-chat/LLaMA-Factory/save/0625/Boiling_point_test/predict/generated_predictions.jsonl'
    base_path = '/zengdaojian/litianle/Mycode/Mol-chat/LLaMA-Factory/save/0625'
    r2 = []
    file_names = os.listdir(base_path)
    for file_name in file_names:
        file_path = base_path + '/' + file_name + "/predict/generated_predictions.jsonl"
        # 读取JSON Lines数据
        data = read_jsonl_file(file_path)
        # 提取标签和预测值，并转换为float类型
        labels = [float(item['label'][:-1]) for item in data]
        predictions = [float(item['predict'][:-1]) for item in data]
        # 计算MAE
        r2_value = calculate_r2(labels, predictions)
        print(f"{file_name} R^2: {r2_value}")
        r2.append(r2_value)
    # 要写入的CSV文件路径
    csv_file_path = '/zengdaojian/litianle/Mycode/Mol-chat/LLaMA-Factory/metric/r2_results.csv'
    # 打开文件，写入模式
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入标题行（如果有的话）
        writer.writerow(['Name', 'R^2'])
        # 写入数据行
        for name, mae in zip(file_names, r2):
            writer.writerow([name, mae])

    print(f"数据已写入 {csv_file_path}")
if __name__ == "__main__":
    main()