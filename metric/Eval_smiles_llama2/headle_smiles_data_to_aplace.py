import csv
import json
import os
import pandas as pd

# 读取文件下的csv文件名
def read_csv_lists(path):
    json_lists = []
    file_lists = os.listdir(path)
    for file_name in file_lists:
        if file_name.endswith('csv'):
            json_lists.append(os.path.join(path, file_name))
    return json_lists

# with open('/zengdaojian/litianle/Mycode/Mol-chat/LLaMA-Factory/data/dataset_info_original.json', 'r', encoding='utf-8') as f:
#     data_info = json.load(f)
base_path = '/zengdaojian/litianle/Mycode/Mol-chat/TSMMG-main/data/eval'
files_path = read_csv_lists(base_path)
name_list = []
for file_path in files_path:
    name = file_path.split('/')[-1].split('.')[0]
    name_list.append(name)
    f_write = open('/zengdaojian/litianle/Mycode/Mol-chat/LLaMA-Factory/metric/Eval_smiles_llama2/data/' + name + '.json', "w", encoding='utf-8')
    datas = pd.read_csv(file_path)
    datas_list = datas['desc'].values.tolist()
    num_id = 1
    results = []
    for data in datas_list:
        item = {"id":num_id, "instruction": str(data), "input": '', "output": '', "history": []}
        results.append(item)
        num_id += 1
    results = json.dumps(results, ensure_ascii=False, indent=4)
    f_write.write(results)
    f_write.close()


print(name_list)
with open('/zengdaojian/litianle/Mycode/Mol-chat/LLaMA-Factory/metric/Eval_smiles_llama2/name.txt', 'w', encoding='utf-8') as file:
    file.write(','.join(name_list))

