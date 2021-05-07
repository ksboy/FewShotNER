CUDA_VISIBLE_DEVICES=0 nohup python -u train_demo.py  \
--train data/structdata/train.txt \
--val data/structdata/dev.txt \
--test data/structdata/support-wnut-5shot/0.txt \
--lr 2e-5 --batch_size 1 \
--trainN 4 --N 6 --K 5 --Q 5 \
--train_iter 30000 \
--val_iter 1000 \
--test_iter 100 \
--val_step 1000 \
--max_length 128 > trainlog/structdata-13-9-5-5.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2,3 nohup python -u train_demo.py  --train data/mydata/train-inter.txt --val data/mydata/val-inter.txt --test data/mydata/test-inter.txt --lr 1e-3 --batch_size 16 --trainN 5 --N 5 --K 1 --Q 1 --train_iter 30000 --val_iter 1000 --test_iter 5000 --val_step 2000 --max_length 100 > trainlog/mydata-inter-5-1.log &
# CUDA_VISIBLE_DEVICES=4,5 nohup python -u train_demo.py  --train data/mydata/train-intra.txt --val data/mydata/val-intra.txt --test data/mydata/test-intra.txt --lr 1e-4 --batch_size 4 --trainN 5 --N 5 --K 5 --Q 5 --train_iter 30000 --val_iter 1000 --test_iter 5000 --val_step 2000 --max_length 100 > trainlog/mydata-intra-5-5.log &
