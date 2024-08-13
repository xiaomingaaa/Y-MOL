import json
import csv
import os
import numpy as np
import pandas as pd


# 读取文件下的json文件名
def read_json_lists(path):
    json_lists = []
    file_lists = os.listdir(path)
    for file_name in file_lists:
        if file_name.endswith('json'):
            json_lists.append(os.path.join(path, file_name))
    return json_lists

def read_file_lists(path):
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


base_path = 'save/mg'
file_lists = read_file_lists(base_path)
names = []
for file_name in file_lists:
    save_path = '../TSMMG-main/outputs/output_gpt2_ft_' + file_name.split('/')[-1].split('eval_')[1] + '.csv'
    names.append(file_name.split('/')[-1].split('eval_')[1])
    f = open(save_path, 'w', encoding='utf-8')
    datas = read_jsonl_file(file_name+'/predict/generated_predictions.jsonl')
    writer = csv.writer(f)
    writer.writerow(['prompt', 'smiles', 'fgs'])
    for data in datas:
        writer.writerow([data['label'], data['predict'], ''])
    print(save_path + '写入完成！！！')
print(names)