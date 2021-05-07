from fewshot_ner_kit.data_loader import get_loader
from fewshot_ner_kit.framework import FewShotNERFramework
from fewshot_ner_kit.word_encoder import BERTWordEncoder
from models.proto import Proto
from models.nnshot import NNShot
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='data/mydata/train-inter.txt',
            help='train file')
    parser.add_argument('--val', default='data/mydata/val-inter.txt',
            help='val file')
    parser.add_argument('--test', default='data/mydata/test-inter.txt',
            help='test file')
    parser.add_argument('--trainN', default=2, type=int,
            help='N in train')
    parser.add_argument('--evalN', default=2, type=int,
            help='N in eval')
    parser.add_argument('--N', default=2, type=int,
            help='N way')
    parser.add_argument('--K', default=2, type=int,
            help='K shot')
    parser.add_argument('--Q', default=3, type=int,
            help='Num of query per class')
    parser.add_argument('--batch_size', default=4, type=int,
            help='batch size')
    parser.add_argument('--train_iter', default=600, type=int,
            help='num of iters in training')
    parser.add_argument('--val_iter', default=100, type=int,
            help='num of iters in validation')
    parser.add_argument('--test_iter', default=500, type=int,
            help='num of iters in testing')
    parser.add_argument('--val_step', default=20, type=int,
           help='val after training how many iters')
    parser.add_argument('--model', default='proto',
            help='model name, must be proto, nnshot, or structshot')
    parser.add_argument('--max_length', default=100, type=int,
           help='max length')
    parser.add_argument('--lr', default=1e-4, type=float,
           help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
           help='weight decay')
    parser.add_argument('--dropout', default=0.0, type=float,
           help='dropout rate')
    parser.add_argument('--grad_iter', default=1, type=int,
           help='accumulate gradient every x iterations')
    parser.add_argument('--load_ckpt', default=None,
           help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
           help='save ckpt')
    parser.add_argument('--fp16', action='store_true',
           help='use nvidia apex fp16')
    parser.add_argument('--only_test', action='store_true',
           help='only test')
    parser.add_argument('--ckpt_name', type=str, default='',
           help='checkpoint name.')


    # only for bert / roberta
    parser.add_argument('--pretrain_ckpt', default=None,
           help='bert / roberta pre-trained checkpoint')

    # only for prototypical networks
    parser.add_argument('--dot', action='store_true', 
           help='use dot instead of L2 distance for proto')

    # only for structshot
    parser.add_argument('--tau', default=0.05, type=float,
           help='StructShot parameter to re-normalizes the transition probabilities')

    # experiment
    parser.add_argument('--use_sgd_for_bert', action='store_true',
           help='use SGD instead of AdamW for BERT.')

    opt = parser.parse_args()
    trainN = opt.trainN
    evalN = opt.evalN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    max_length = opt.max_length
    
    print("{}-way-{}-shot Few-Shot NER".format(N, K))
    print("model: {}".format(model_name))
    print("max_length: {}".format(max_length))
    
    pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
    word_encoder = BERTWordEncoder(
            pretrain_ckpt,
            max_length)

    train_data_loader = get_loader(opt.train, word_encoder,
            N=trainN, K=K, Q=Q, batch_size=batch_size, max_length=max_length)
    val_data_loader = get_loader(opt.val, word_encoder,
            N=evalN, K=K, Q=Q, batch_size=batch_size, max_length=max_length)
    test_data_loader = get_loader(opt.test, word_encoder,
            N=N, K=K, Q=Q, batch_size=batch_size, max_length=max_length)

        
    prefix = '-'.join([model_name, opt.train.split('/')[-1], opt.val.split('/')[-1], str(N), str(K)])
    if opt.dot:
        prefix += '-dot'
    if len(opt.ckpt_name) > 0:
        prefix += '-' + opt.ckpt_name
    
    if model_name == 'proto':
        print('use proro')
        model = Proto(word_encoder, dot=opt.dot)
        framework = FewShotNERFramework(train_data_loader, val_data_loader, test_data_loader)
    elif model_name == 'nnshot':
        print('use nnshot')
        model = NNShot(word_encoder, dot=opt.dot)
        framework = FewShotNERFramework(train_data_loader, val_data_loader, test_data_loader)
    elif model_name == 'structshot':
        print('use structshot')
        model = NNShot(word_encoder, dot=opt.dot)
        framework = FewShotNERFramework(train_data_loader, val_data_loader, test_data_loader, N=opt.N, tau=opt.tau, train_fname=opt.train, viterbi=True)
    else:
        raise NotImplementedError
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt
    print('model-save-path:', ckpt)

    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test:
        if opt.lr == -1:
            opt.lr = 2e-5

        framework.train(model, prefix, batch_size, trainN, evalN, K, Q,
                load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                val_step=opt.val_step, fp16=opt.fp16,
                train_iter=opt.train_iter, warmup_step=int(opt.train_iter * 0.1), val_iter=opt.val_iter, learning_rate=opt.lr, use_sgd_for_bert=opt.use_sgd_for_bert)
    else:
        ckpt = opt.load_ckpt
        if ckpt is None:
            print("Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint.")
            ckpt = 'none'

    precision_total = 0.0
    recall_total = 0.0
    f1_total = 0.0
    fp_total = 0.0
    fn_total = 0.0
    within_total = 0.0
    outer_total = 0.0
    if not opt.only_test:
        n = 1
    else:
        n = 5
    for i in range(n):
        precision, recall, f1, fp, fn, within, outer = framework.eval(model, batch_size, N, K, Q, opt.test_iter, ckpt=ckpt)
        precision_total += precision
        recall_total += recall
        f1_total += f1
        fp_total += fp
        fn_total += fn
        within_total += within
        outer_total += outer
    print("RESULT: precision: %.4f, recall: %.4f, f1:%.4f" % (precision_total / n, recall_total / n, f1_total / n))
    print('ERROR ANALYSIS: fp: %.4f, fn: %.4f, within:%.4f, outer: %.4f'%(fp_total / n, fn_total / n, within_total / n, outer_total / n))

if __name__ == "__main__":
    main()
