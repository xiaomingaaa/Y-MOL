from transformers import AutoTokenizer
import transformers
import torch
import json
import os
import argparse
from tqdm import tqdm
import time
import re
import warnings
warnings.filterwarnings('ignore')

def read_json_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def find_first_number(string):
    # 正则表达式匹配整数或小数
    pattern = r'\b\d+(?:\.\d+)?\b'
    match = re.search(pattern, string)
    if match:
        # 返回找到的第一个数值
        return float(match.group())
    else:
        # 如果没有找到数值，返回None
        return ''

# 读取文件下的json文件名
def read_json_lists(path):
    json_lists = []
    file_lists = os.listdir(path)
    for file_name in file_lists:
        if file_name.endswith('json'):
            json_lists.append(os.path.join(path, file_name))
    return json_lists

def write_arrays_to_jsonl(array1, array2, filename):
    """
    将两个数组写入JSONL文件。
    :param array1: 第一个数组
    :param array2: 第二个数组
    :param filename: 输出的JSONL文件名
    """
    # 检查两个数组长度是否相同
    if len(array1) != len(array2):
        raise ValueError("两个数组长度必须相同")
    # 打开文件用于写入
    with open(filename, 'w') as outfile:
        # 遍历数组元素
        for item1, item2 in zip(array1, array2):
            # 将元素转换为字典
            json_object = {'label': item1, 'predict': item2}
            # 将字典写入文件，每个字典占一行
            json.dump(json_object, outfile)
            outfile.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default='/zengdaojian/litianle/Mycode/Mol-chat/LLaMA-Factory/finetune_model/llama2-7B-hf',type=str)
    parser.add_argument(
        "--data_folder",
        type=str,
        default='/zengdaojian/litianle/Mycode/Mol-chat/LLaMA-Factory/datasets/0625_test'
    )
    args = parser.parse_args()
    
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model_name_or_path,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # 加载数据
    files_name = read_json_lists(args.data_folder)
    for name in files_name:
        # 写入的文件
        f = open('/zengdaojian/litianle/Mycode/Mol-chat/LLaMA-Factory/metric/Property/' + name.split('/')[-1].split('.')[0] + '.jsonl', 'w', encoding='utf-8')
        datas = read_json_data(name)
        for data in tqdm(datas):
            # data = datas[i]
            result = {}
            inputs =  data['Q'] + ' Answer a four-digit decimal place of -10 to 10: '
            # inputs =  data['Q'] + ' Answer a numeric value: '
            sequences = pipeline(
                inputs,
                do_sample=False,
                temperature=0.001,
                top_k=1,
                top_p=0.9,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=150
            )
            for seq in sequences:
                output = seq['generated_text'].split('Answer a four-digit decimal place of -10 to 10:')[1]
            print(find_first_number(output))
            result['label'] = data['A']
            result['predict'] = str(find_first_number(output))
            f.write(json.dumps(result) + '\n')
        f.close()