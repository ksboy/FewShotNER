config_json=deepspeed_config.json

OUTPUT_DIR=output/ontonotes-\>wnut
MODEL=structshot

# --train data/structdata/ontonotes-5.0/train.txt \
# --val data/structdata/ontonotes-5.0/dev.txt \

mkdir -p $OUTPUT_DIR
# CUDA_VISIBLE_DEVICES=0 nohup python -m debugpy --listen 0.0.0.0:8888 --wait-for-client train_demo.py \
CUDA_VISIBLE_DEVICES=0 nohup python -u train_demo.py  \
--model $MODEL \
--train data/structdata/conll-2003/train.txt \
--val data/structdata/conll-2003/dev.txt \
--test data/structdata/wnut/test.txt \
--test_support data/structdata/support-wnut-5shot/0.txt \
--test_query data/structdata/wnut/test.txt \
--support_fixed \
--lr 2e-5 --batch_size 1 \
--trainN 4 --evalN 4 --N 6 --K 5 --Q 5 \
--fp16 \
--train_iter 30000 \
--val_iter 1000 \
--test_iter 100 \
--val_step 1000 \
--output_dir $OUTPUT_DIR \
--max_length 128 > $OUTPUT_DIR/$MODEL.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2,3 nohup python -u train_demo.py  --train data/mydata/train-inter.txt --val data/mydata/val-inter.txt --test data/mydata/test-inter.txt --lr 1e-3 --batch_size 16 --trainN 5 --N 5 --K 1 --Q 1 --train_iter 30000 --val_iter 1000 --test_iter 5000 --val_step 2000 --max_length 100 > trainlog/mydata-inter-5-1.log &
# CUDA_VISIBLE_DEVICES=4,5 nohup python -u train_demo.py  --train data/mydata/train-intra.txt --val data/mydata/val-intra.txt --test data/mydata/test-intra.txt --lr 1e-4 --batch_size 4 --trainN 5 --N 5 --K 5 --Q 5 --train_iter 30000 --val_iter 1000 --test_iter 5000 --val_step 2000 --max_length 100 > trainlog/mydata-intra-5-5.log &
# --only_test \
# --load_ckpt ./checkpoint/proto-train.txt-dev.txt-6-5.pth.tar \

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