#!/bin/bash
# arr=('ddi_deng_instructions_samples_test1' 'ddi_ryu_instructions_samples_test1' 'dti_drugbank_instructions_test1' 'dti_drugcentral_instructions_test1')
arr=('eval_drd2_qed_sa' 'eval_drd2_qed' 'eval_sa' 'eval_logp_-3' 'eval_hia_v2' 'eval_bbb_qed' 'eval_gsk3_qed_sa' 'eval_hia' 'eval_logp_-1' 'eval_gsk3_qed' 'eval_hia_qed_sa' 'eval_hia_v1' 'eval_logp_5' 'eval_furan_qed' 'eval_qed' 'eval_benzene' 'eval_logp_3' 'eval_logp_1' 'eval_drd2_v1' 'eval_drd2' 'eval_gsk3' 'eval_bbb_qed_sa' 'eval_bbb_v1' 'eval_2fg' 'eval_drd2_v2' 'eval_bbb' 'eval_furan' 'eval_gsk3_v2' 'eval_gsk3_v1' 'eval_hia_qed' 'eval_bbb_v2' 'eval_benzene_qed')

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
        --output_dir save/mg/$str/predict \
        --overwrite_cache \
        --overwrite_output_dir \
        --cutoff_len 2048 \
        --preprocessing_num_workers 60 \
        --per_device_eval_batch_size 1 \
        --predict_with_generate 
done

