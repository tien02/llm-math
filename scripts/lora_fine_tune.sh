BASE_DIR=/path/to/root/dir
MODEL=EleutherAI/llemma_7b
CONFIG=${BASE_DIR}/script/zero3_offload.json
OUTDIR=${BASE_DIR}/ckpt/${MODEL}
TRAIN_FILE=${BASE_DIR}/data/math_train.json
BATCHSIZE=1

deepspeed --include localhost:0,1,2,3  ${BASE_DIR}/train_math.py \
    --deepspeed ${CONFIG} \
    --lora_enable \
    --lora_r 128 \
    --lora_alpha 256 \
    --model_name_or_path ${MODEL} \
    --data_path ${TRAIN_FILE} \
    --dataloader_num_workers 8 \
    --fp16 \
    --output_dir ${OUTDIR} \
    --per_device_train_batch_size ${BATCHSIZE} \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 4 \
    --save_strategy "steps" \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --logging_dir "$OUTDIR" \
    --report_to wandb \