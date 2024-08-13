import argparse
import json
import torch

'''
orig_data: {"instruction": "题目：小明买了一支钢笔，花费了5元，又买了一本书，花费8元，现在他手里还有10元钱，他手上原来有多少钱？", "input": "", "output": "\n令小明手上原来有的钱为X元。根据题目描述，得出以下方程式：\nX - 5 - 8 = 10\n化简可得：\nX = 23\n因此，小明手上原来有23元钱。"}
convert: {"instruction": "题目：小明买了一支钢笔，花费了5元，又买了一本书，花费8元，现在他手里还有10元钱，他手上原来有多少钱？", "input": "", "output": "\n令小明手上原来有的钱为X元。根据题目描述，得出以下方程式：\nX - 5 - 8 = 10\n化简可得：\nX = 23\n因此，小明手上原来有23元钱。", "history":[]}

'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orig_data",
        type=str,
        default='/zengdaojian/litianle/Mycode/Mol-chat/LLaMA-Factory/inference/test6.6/TPSA_QA_result.json'
    )
    parser.add_argument(
        "--write_data",
        type=str,
        default='/zengdaojian/litianle/Mycode/Mol-chat/LLaMA-Factory/inference/test6.6/TPSA_QA_result_predict.json'
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='Mol'
    )
    args = parser.parse_args()
    f_write = open(args.write_data,"w", encoding='utf-8')
    results = []
    with open(args.orig_data) as f:
        datas = json.load(f)
        num_id = 1
        for data in datas:
            # conversations = [{"from": "human", "value": str(data['instruction'])+str(data['input'])},{"from": "assistant", "value": str(data['output'])}]
            # conversations = [{"from": "human", "value": data['input']},{"from": "assistant", "value": data['target']}]
            uniq_id = data['id'] if "id" in data else args.dataset_name+"-"+str(num_id)
            item = {"id":uniq_id, "instruction": str(data['Q']), "input": '', "output": str(data['A']), "history": []}
            results.append(item)
            # results.append(json.dumps(item, ensure_ascii=False, indent=4))
            # f_write.write(json.dumps(item, ensure_ascii=False, indent=4)+"\n")
            num_id += 1
            # f_write.close()
        results = json.dumps(results, ensure_ascii=False, indent=4)
        f_write.write(results)
    f_write.close()


if __name__ == "__main__":
    main()