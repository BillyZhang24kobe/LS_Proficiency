# WAND_NAME="train_vicuna_13b_lr_1e_5_swords" CUDA_VISIBLE_DEVICES=5,7 torchrun --nproc_per_node=2 --master_port=20003 fastchat/train/train_mem.py \
#     --model_name_or_path "lmsys/vicuna-13b-v1.5" \
#     --data_path data/train/train_swords_filtered.csv  \
#     --bf16 True \
#     --output_dir /local/data/xuanming/models/output_vicuna_13b_train_swords_lr_1e_5 \
#     --num_train_epochs 5 \
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
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'

WAND_NAME="train_flant5" CUDA_VISIBLE_DEVICES=3 python fastchat/train/train_flant5.py \
    --model_name_or_path "google/flan-t5-xl" \
    --data_path data/train/train_trial_101_final.csv \
    --bf16 True \
    --output_dir /local/data/xuanming/models/output_flant5_xl_trial_lr_1e_4 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 1200 \
    --save_total_limit 10 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_steps 100 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess False \