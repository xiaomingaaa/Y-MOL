from transformers import AutoTokenizer
import transformers
import torch
import json
import os
import argparse
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

def read_json_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

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
    parser.add_argument('--data_name', default='ddi_deng_instructions_samples', type=str)
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
    f = open('/zengdaojian/litianle/Mycode/Mol-chat/LLaMA-Factory/metric/DDI/' + args.data_name + '.jsonl', 'w', encoding='utf-8')
    datas = read_json_data('/zengdaojian/litianle/Mycode/Mol-chat/LLaMA-Factory/data/' + args.data_name + '.json')
    start_time = time.time()
    for data in tqdm(datas):
        # data = datas[i]
        result = {}
        inputs = 'Answer yes or no, ' + data['input']
        sequences = pipeline(
            inputs,
            do_sample=False,
            temperature=0.001,
            top_k=1,
            top_p=0.9,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=100
        )
        for seq in sequences:
            output = seq['generated_text'].split('?')[1]
        if 'yes' or 'Yes' in output:
            result['label'] = 'Yes'
        elif 'no' or 'No' in output:
            result['label'] = 'No'
        else:
            result['label'] = ''
        result['predict'] = data['output']
        f.write(json.dumps(result) + '\n')
    f.close()