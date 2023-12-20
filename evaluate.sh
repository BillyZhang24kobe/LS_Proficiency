# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=20003 evaluate.py \
#     --model_name_or_path "/local/data/xuanming/models/output_vicuna_13b_trial_lr_1e_5/checkpoint-306" \
#     --data_path data/dev/LS-Pro_test_trial_61_final.csv \
#     --bf16 True \
#     --output_dir /local/data/xuanming/models/output_vicuna_13b_trial_lr_1e_5 \
#     --num_train_epochs 10 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "epoch" \
#     --save_steps 1200 \
#     --save_total_limit 10 \
#     --learning_rate 5e-5 \
#     --weight_decay 0. \
#     --warmup_steps 100 \
#     --lr_scheduler_type "linear" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --lazy_preprocess False \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'

CUDA_VISIBLE_DEVICES=3 python3 evaluate.py \
    --model_name_or_path "/local/data/xuanming/models/output_vicuna_13b_trial_lr_1e_5/checkpoint-357" \
    --data_path data/dev/LS-Pro_test_trial_68_final.csv \
    --bf16 True \
    --output_dir outputs/vicuna-7b-v1.5_trial \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess False \
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \