#!/bin/bash
arr=('Flexibility_test' 'pKa_acidic_test' 'TPSA_test' 'LogS_test' 'Volume_test' 'pKa_basic_test' 'nRot_test' 'nHet_test' 'LogD7_4_test' 'nHA_test' 'nRing_test' 'Molecular_Weight_test' 'nRig_test' 'Melting_point_test' 'fChar_test' 'nHD_test' 'Stereo_Centers_test' 'Boiling_point_test' 'MaxRing_test' 'Density_test' 'LogP_test')

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
        --output_dir save/mpp/$str/predict \
        --overwrite_cache \
        --overwrite_output_dir \
        --cutoff_len 1024 \
        --preprocessing_num_workers 60 \
        --per_device_eval_batch_size 1 \
        --predict_with_generate 
done

