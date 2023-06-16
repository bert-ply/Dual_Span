# -*- coding: utf-8 -*-
# @Time : 2023/2/17 15:50
# @Author : Pan Li

import json, os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import time
import sys
sys.path.append('../')
from vocab import Vocab
from Data_process import ABSADataLoader, load_pretrained_embedding
from Common.Glod_labels import gold_labels
from Common.Loss import log_likelihood
from Common.metrics import SpanEvaluator
from Model import Synastemodel

def train(args):
    train_sentence_packs = json.load(open(args.data_path + "train.json"))
    # print(train_sentence_packs[0:5])
    dev_sentence_packs = json.load(open(args.data_path + "dev.json"))
    test_sentence_packs = json.load(open(args.data_path + "test.json"))

    token_vocab = Vocab.load_vocab(args.data_path + 'vocab_tok.vocab')  # token嵌入
    post_vocab = Vocab.load_vocab(args.data_path + 'vocab_post.vocab')     # 位置嵌入
    deprel_vocab = Vocab.load_vocab(args.data_path + 'vocab_deprel.vocab')  # 依存边嵌入
    postag_vocab = Vocab.load_vocab(args.data_path + 'vocab_postag.vocab')  # 词性嵌入
    vocab = (token_vocab, post_vocab, deprel_vocab, postag_vocab)
    args.tok_size = len(token_vocab)
    args.post_size = len(post_vocab)  # 相对位置
    args.deprel_size = len(deprel_vocab)  # 依存关系索引
    args.postag_size = len(postag_vocab)  # 词性标注索引
    print(
        "token_vocab: {},post_vocab: {}, deprel_vocab: {}, postag_vocab: {}".format(
            len(token_vocab), len(post_vocab), len(deprel_vocab), len(postag_vocab)
        )
    )

    args.tok_size = len(token_vocab)
    args.post_size = len(post_vocab)
    args.postag_size = len(postag_vocab)
    args.dep_size = len(deprel_vocab)
    # load pretrained word emb
    print("Loading pretrained word emb...")
    word_emb = load_pretrained_embedding(glove_dir=args.glove_dir, word_list=token_vocab.itos)
    assert len(word_emb) == len(token_vocab)
    assert len(word_emb[0]) == args.emb_dim
    word_emb = torch.FloatTensor(word_emb)  # convert to tensor


    trainset = ABSADataLoader(train_sentence_packs, args.batch_size, vocab, args, shuffle=True)
    devset = ABSADataLoader(dev_sentence_packs, args.batch_size, vocab, args, shuffle=False)
    testset = ABSADataLoader(test_sentence_packs, args.batch_size, vocab, args, shuffle=False)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    model = Synastemodel(args, emb_matrix=word_emb)
    model.cuda()

    parameters = [p for p in model.parameters() if p.requires_grad]
    print("Building Optimizer...")
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)

    # training
    metric = SpanEvaluator()

    best_joint_f1 = 0
    best_joint_epoch = 0
    start = time.clock()
    for epoch in range(1, args.epochs + 1):
        best_joint_epoch += 1
        print('Epoch:{}'.format(epoch) + "-" * 60)
        train_loss, train_step = 0.0, 0
        model.train()
        for i, batch in enumerate(trainset):
            batch = [b.cuda() for b in batch]
            # tok, postag, head, deprel, spans, span_labels, count, relations, relation_labels, postag_special, length
            input = batch[0:11]

            # forward
            spans_probability, span_indices, relations_probability, candidate_indices = model(input)

            gold_span_indices, gold_span_labels = gold_labels(span_indices, batch[4], batch[5])

            # span loss
            loss_entity = log_likelihood(spans_probability, span_indices, gold_span_indices, gold_span_labels)

            gold_relation_indices, gold_relation_labels = gold_labels(candidate_indices, batch[7], batch[8])
            loss_relation = log_likelihood(relations_probability, candidate_indices, gold_relation_indices, gold_relation_labels)
            # loss compute
            loss = 0.2 * loss_entity + loss_relation
            train_loss += loss
            train_step += 1

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if train_step % 20 == 0:
                print(
                    "{}/{} train_loss: {:.6f}".format(
                        i, len(trainset), train_loss / train_step
                    )
                )


        val_precision, val_recall, val_f1 = evaluate(model, metric, devset)
        print(
            "End of {} Evaluation val_precision: {:.4f}, val_recall: {:.4f}, val_F1: {:.4f}".format
            (epoch, val_precision, val_recall, val_f1))
        if val_f1 > best_joint_f1:
            print(
                f"best F1 performence has been updated: {best_joint_f1:.5f} --> {val_f1:.5f}"
            )
            if not os.path.exists(args.model_dir):
                os.makedirs(args.model_dir)
            model_path = args.model_dir + 'model.pt'
            torch.save(model, model_path)
            best_joint_f1 = val_f1
            best_joint_epoch = epoch
            print("new best model saved.")
    print('best epoch: {}\tbest dev f1: {:.5f}\n\n'.format(best_joint_epoch, best_joint_f1))
    print("Evaluation on testset:")
    end = time.clock()
    runTime = (end - start) / args.epochs
    print("time：", runTime)
    model_path = args.model_dir + 'model.pt'
    model = torch.load(model_path).cuda()
    model.eval()
    # ATE_P, ATE_R, ATE_F, OTE_P, OTE_R, OTE_F = span_evaluate(model, metric, testset)
    # print("-----------------------------")
    # print("Evaluation ATE_Precision: %.5f | ATE_Recall: %.5f | ATE_F1: %.5f, OTE_Precision: %.5f | OTE_Recall: %.5f | OTE_F1: %.5f" %
    #       (ATE_P, ATE_R, ATE_F, OTE_P, OTE_R, OTE_F))
    test_precision, test_recall, test_f1 = evaluate(model, metric, testset)
    print("-----------------------------")
    print("Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f" %
          (test_precision, test_recall, test_f1))
    data = [[args.seed, test_recall, test_recall, test_f1]]
    data = pd.DataFrame(data)
    data.to_csv(args.model_dir + 'result.csv', mode='a', index=False, header=False)

def evaluate(model, metric, dataset):
    model.eval()
    metric.reset()
    with torch.no_grad():
        val_loss, val_step = 0.0, 0
        for i, batch in enumerate(dataset):
            batch = [b.cuda() for b in batch]
            # tok, postag, head, deprel, spans, span_labels, count, relations, relation_labels, postag_special length
            input = batch[0:11]

            spans_probability, span_indices, relations_probability, candidate_indices = model(input)

            gold_span_indices, gold_span_labels = gold_labels(span_indices, batch[4], batch[5])
            gold_relation_indices, gold_relation_labels = gold_labels(candidate_indices, batch[7], batch[8])

            num_correct, num_infer, num_label = metric.compute(relations_probability.cpu(),
                                                               torch.tensor(gold_relation_labels))
            metric.update(num_correct, num_infer, num_label)
    precision, recall, f1 = metric.accumulate()
    model.train()
    return precision, recall, f1

def evaluate_span(y_pred, label, N):
    P, R, F = 0, 0, 0
    prediciton_list = torch.zeros(len(y_pred))
    reference_list = torch.zeros(len(y_pred))
    if N == 1:
        for i in range(len(label)):
            if label[i] == 1:
                reference_list[i] = 1
            if y_pred[i] == 1:
                prediciton_list[i] = 1
        TP = torch.logical_and(reference_list == prediciton_list, prediciton_list != 0).sum().item()  # 预测为1，真实为1
        FP = (prediciton_list != 0).sum().item()  # 预测为1的
        FN = (reference_list != 0).sum().item()  # 真实为1

        if FP != 0 and FN != 0:
            P = TP/FP
            R = TP/FN
            F = 2 * P * R / (P + R)
        elif FP == 0 and FN != 0:
            P = 0
            R = TP/FN
            F = 2 * P * R / (P + R)
        elif FN == 0 and FP != 0:
            R = 0
            P = TP / FP
            F = 2 * P * R / (P + R)

    elif N == 2:
        for i in range(len(label)):
            if label[i] == 2:
                reference_list[i] = 1
            if y_pred[i] == 2:
                prediciton_list[i] = 1
        TP = torch.logical_and(reference_list == prediciton_list, prediciton_list != 0).sum().item()  # 预测为1，真实为1
        FP = (prediciton_list != 0).sum().item()  # 预测为1的
        FN = (reference_list != 0).sum().item()  # 真实为1

        if FP != 0 and FN != 0:
            P = TP / FP
            R = TP / FN
            F = 2 * P * R / (P + R)
        elif FP == 0 and FN != 0:
            P = 0
            R = TP / FN
            F = 2 * P * R / (P + R)
        elif FN == 0 and FP != 0:
            R = 0
            P = TP / FP
            F = 2 * P * R / (P + R)
    return P, R, F

def span_evaluate(model, metric, dataset):
    model.eval()
    metric.reset()

    with torch.no_grad():
        y_pred = []
        labels = []
        val_loss, val_step = 0.0, 0
        for i, batch in enumerate(dataset):
            batch = [b.cuda() for b in batch]
            # tok, postag, head, deprel, spans, span_labels, count, relations, relation_labels, postag_special, length
            input = batch[0:11]

            spans_probability, span_indices, relations_probability, candidate_indices = model(input)
            gold_span_indices, gold_span_labels = gold_labels(span_indices, batch[4], batch[5])

            pred = torch.squeeze(spans_probability.argmax(-1),dim=0)
            # print(pred)
            # print(gold_span_labels)
            labels += gold_span_labels
            y_pred += pred.cpu().numpy().tolist()
        # print(y_pred)
        # print(labels)
    ATE_P, ATE_R, ATE_F = evaluate_span(y_pred, labels, 1)
    OTE_P, OTE_R, OTE_F = evaluate_span(y_pred, labels, 2)
    model.train()
    return ATE_P, ATE_R, ATE_F, OTE_P, OTE_R, OTE_F

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str, default="./savemodel/res16/", help='model path prefix') #数据保存路径
    parser.add_argument("--data_path", default="../Data/res16/", type=str, help="The path of train set.")
    parser.add_argument('--glove_dir', type=str, default="../Data/", help='model path prefix') #数据保存路径
    parser.add_argument("--max_seq_len", default=128, type=int,
                        help="The maximum input sequence length. Sequences longer than this will be split automatically.")
    parser.add_argument('--emb_dim', type=int, default=300, help='dimension of word')
    parser.add_argument("--fnn_hidden_dim", type=int, default=150, help="hidden dim.")

    parser.add_argument("--input_dropout", type=float, default=0.7, help="Input dropout rate.")  # 输入dropout
    parser.add_argument('--bidirect', default=True, help='Do use bi-RNN layer.')
    parser.add_argument('--rnn_hidden', type=int, default=300, help='RNN hidden state size.')
    parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
    parser.add_argument('--rnn_dropout', type=float, default=0.5, help='RNN dropout rate.')

    parser.add_argument("--span_pruned_threshold", type=float, default=0.5, help="threshold hyper-parameter for span pruned.")
    parser.add_argument('--span_maximum_length', type=int, default=8, help='The maximum span length.')
    parser.add_argument('--span_width_dim', type=int, default=30, help='dimension of span width')
    parser.add_argument('--triplet_distance_dim', type=int, default=128, help='dimension of triplet distance')
    # parser.add_argument('--post_dim', type=int, default=30, help='Position embedding dimension.')
    parser.add_argument('--postag_dim', type=int, default=30, help='dimension of postag_dim')
    parser.add_argument('--dep_dim', type=int, default=30, help='dimension of deprel')
    parser.add_argument("--use_dep_span", default=True, help='use deprel extract span')
    parser.add_argument("--sapn_use_dep", default=True, help='use span deprel or not use')
    parser.add_argument("--triplet_use_dep", default=True, help='use triplet deprel or not use')

    parser.add_argument('--batch_size', type=int, default=1, help='bath size')
    parser.add_argument('--epochs', type=int, default=60, help='training epoch number')
    parser.add_argument('--class_num', type=int, default=3, help='aspect/opinion class')
    parser.add_argument('--sentiment_class', type=int, default=4, help='sentiment class')

    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--emb_dropout', type=float, default=0.7)                       #嵌入dropout
    parser.add_argument('--lr', default=0.001, type=float)                              #学习率
    parser.add_argument('--device', type=str, default="cpu", help='gpu or cpu')         #选取设备

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    train(args)