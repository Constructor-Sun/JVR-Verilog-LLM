NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model models/Qwen3-8B \
    --resume_from_checkpoint models/sft-8b/v2-20260201-132917/checkpoint-1500 \
    --train_type full \
    --dataset 'data/training_data/integral/codev_r1_sft_mixed_filtered_chatml.parquet' \
    --torch_dtype bfloat16 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 32 \
    --gradient_checkpointing true \
    --packing true \
    --save_steps 100 \
    --logging_steps 5 \
    --max_length 16384 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 15 \
    --lr_scheduler_type cosine \
    --output_dir models/sft-8b \
    --use_liger_kernel true \
    --attn_impl flash_attn \
    --report_to wandb \
    --run_name "qwen3-8b-sft-2gpu" \
    --deepspeed zero3 \