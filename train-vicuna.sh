WAND_NAME="train_vicuna_13b_lr_1e_5_combined" CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=20003 fastchat/train/train_mem.py \
    --model_name_or_path "lmsys/vicuna-13b-v1.5" \
    --data_path data/train/train_combined_2298.csv  \
    --bf16 True \
    --output_dir /local/data/xuanming/models/output_vicuna_13b_train_combined_lr_1e_5 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 1200 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_steps 100 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess False \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'

# WAND_NAME="train_lr_1e_5_combined" CUDA_VISIBLE_DEVICES=0 python fastchat/train/train_mem.py \
#     --model_name_or_path "lmsys/vicuna-7b-v1.5" \
#     --data_path data/train/train_combined_2298.csv \
#     --bf16 True \
#     --output_dir /local/data/xuanming/models/output_vicuna_7b_train_combined_lr_1e_5 \
#     --num_train_epochs 10 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "epoch" \
#     --save_steps 1200 \
#     --save_total_limit 10 \
#     --learning_rate 1e-5 \
#     --weight_decay 0. \
#     --warmup_steps 100 \
#     --lr_scheduler_type "linear" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --lazy_preprocess False \