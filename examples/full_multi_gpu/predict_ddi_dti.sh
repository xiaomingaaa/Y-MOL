#!/bin/bash
arr=('ddi_deng_instructions_samples_test1' 'ddi_ryu_instructions_samples_test1' 'dti_drugbank_instructions_test1' 'dti_drugcentral_instructions_test1')

for str in "${arr[@]}"
do
    pgrep -f python | xargs kill -9
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
        --config_file examples/accelerate/single_config.yaml \
        src/train.py \
        --stage sft \
        --do_predict \
        --model_name_or_path save_model/test \
        --dataset $str \
        --dataset_dir data \
        --template default \
        --finetuning_type full \
        --output_dir save/ddi_dti/$str/predict \
        --overwrite_cache \
        --overwrite_output_dir \
        --cutoff_len 4096 \
        --preprocessing_num_workers 60 \
        --per_device_eval_batch_size 1 \
        --predict_with_generate
done

