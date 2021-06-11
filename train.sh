OUTPUT_DIR=output/conll-\>wnut
MODEL=proto

# --train data/structdata/ontonotes-5.0/train.txt \
# --val data/structdata/ontonotes-5.0/dev.txt \

mkdir -p $OUTPUT_DIR
# CUDA_VISIBLE_DEVICES=0 nohup python -u train_demo.py  \
# CUDA_VISIBLE_DEVICES=0 nohup python -m debugpy --listen 0.0.0.0:9999 --wait-for-client train_demo.py \
CUDA_VISIBLE_DEVICES=0 nohup python -u train_demo.py  \
--model $MODEL \
--pretrain_ckpt bert-base-uncased \
--train data/structdata/conll-2003/train.txt \
--val data/structdata/conll-2003/dev.txt \
--test data/structdata/wnut/test.txt \
--test_support data/structdata/support-wnut-5shot/0.txt \
--test_query data/structdata/wnut/test.txt \
--support_fixed \
--query_fixed \
--lr 2e-5 --batch_size 1 \
--trainN 4 --evalN 4 --N 6 --K 5 --Q 5 \
--train_iter 30000 \
--val_iter 1000 \
--test_iter 100 \
--val_step 1000 \
--output_dir $OUTPUT_DIR \
--only_test \
--load_ckpt ./output/conll-\>wnut/checkpointepoch=2.ckpt \
--save_ckpt $OUTPUT_DIR/$MODEL\_support_query_fixed.pth.tar \
--max_length 128 > $OUTPUT_DIR/$MODEL\_from_ckpt.log 2>&1 &
# --fp16 \


# CUDA_VISIBLE_DEVICES=0 nohup python -u train_demo.py  \
# --model structshot \
# --train data/structdata/ontonotes-5.0/train.txt \
# --val data/structdata/ontonotes-5.0/dev.txt \
# --test data/structdata/test-wnut.txt \
# --lr 2e-5 --batch_size 1 \
# --trainN 4 --evalN 4 --N 6 --K 5 --Q 5 \
# --train_iter 30000 \
# --val_iter 1000 \
# --test_iter 100 \
# --val_step 1000 \
# --max_length 128 > trainlog/structdata-structshot-4-6-5-5.log 2>&1 &