config_json=deepspeed_bsz24_config.json
deepspeed --num_nodes 1 --num_gpus 1 \
       --master_port=${MASTER_PORT} \
       --hostfile ${HOSTFILE} \
       nvidia_run_squad_deepspeed.py \
       --bert_model bert-large-uncased \
       --do_train \
       --do_lower_case \
       --predict_batch_size 3 \
       --do_predict \
       --train_batch_size 4 \
       --predict_batch_size 3 \
       --learning_rate ${LR} \
       --num_train_epochs 2.0 \
       --max_seq_length 384 \
       --doc_stride 128 \
       --output_dir $OUTPUT_DIR \
       --job_name ${JOB_NAME} \
       --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
       --fp16 \
       --deepspeed \
       --deepspeed_config ${config_json} \
       --dropout ${DROPOUT} \
       --model_file $MODEL_FILE \
       --seed ${SEED} \
       --preln \