#!/bin/bash
arr=('ddi_deng_instructions_samples_test1' 'ddi_ryu_instructions_samples_test1' 'dti_drugbank_instructions_test1' 'dti_drugcentral_instructions_test1')

for str in "${arr[@]}"
do
    pgrep -f python | xargs kill -9
    python /zengdaojian/litianle/Mycode/Mol-chat/LLaMA-Factory/inference/llama2_inference_ddi.py \
        --data_name $str &> /zengdaojian/litianle/Mycode/Mol-chat/LLaMA-Factory/metric/DDI/$str.log
done

