#!/bin/bash
export HF_DATASETS_CACHE=''
pgrep -f python | xargs kill -9

deepspeed --num_gpus 1 src/train.py \
    --deepspeed examples/deepspeed/ds_z2_config.json \
    --stage sft \
    --do_train \
    --model_name_or_path finetune_model/llama2-7B-hf \
    --dataset 0625,ddi_deng_instructions_samples_train1,ddi_ryu_instructions_samples_train1,dti_drugbank_instructions_train1,dti_drugcentral_instructions_train1,zp \
    --dataset_dir data \
    --template default \
    --finetuning_type full \
    --output_dir save_model/test \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 2000000 \
    --eval_steps 2000000 \
    --evaluation_strategy steps \
    --learning_rate 4e-5 \
    --num_train_epochs 6.0 \
    --val_size 0.001 \
    --ddp_timeout 180000000 \
    --plot_loss \
    --report_to none \
    --bf16 &> examples/full_multi_gpu/test.log
