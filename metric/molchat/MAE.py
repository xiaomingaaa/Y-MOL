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
    base_path = 'save/mg'
    MAE = []
    file_names = os.listdir(base_path)
    for file_name in file_names:
        file_path = base_path + '/' + file_name + "/predict/generated_predictions.jsonl"
        # 读取JSON Lines数据
        data = read_jsonl_file(file_path)
        # 提取标签和预测值，并转换为float类型
        labels = [float(item['label'][:-1]) for item in data]
        predictions = [float(item['predict'][:-1]) for item in data]
        # 计算MAE
        mae_value = calculate_mae(labels, predictions)
        print(f"{file_name} MAE: {mae_value}")
        MAE.append(mae_value)
    # 要写入的CSV文件路径
    csv_file_path = 'metric/molchat/mae_results.csv'
    # 打开文件，写入模式
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # 写入标题行（如果有的话）
        writer.writerow(['Name', 'MAE'])
        
        # 写入数据行
        for name, mae in zip(file_names, MAE):
            writer.writerow([name, mae])

    print(f"数据已写入 {csv_file_path}")
if __name__ == "__main__":
    main()