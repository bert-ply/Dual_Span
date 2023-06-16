# -*- coding: utf-8 -*-
# @Time : 2022/12/28 16:40
# @Author : Pan Li
import math
import torch
import numpy as np
from Common.Tree_glove import head_to_adj, postag_adj, Span_create, postag_Span_create
import json
from Vocab import Vocab
import pandas as pd
import time
from datetime import datetime

starttime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(starttime)
start = time.perf_counter()

spans2id = {'invalid': 0, 'aspect': 1, 'opinion': 2}
sentiment2id = {'invalid': 0, 'negative': 1, 'neutral': 2, 'positive': 3}

data_path = './Data/D2/res16/'
train_sentence_packs = json.load(open(data_path + "train.json", encoding='UTF-8'))
    # print(train_sentence_packs[0:5])
dev_sentence_packs = json.load(open(data_path + "dev.json", encoding='UTF-8'))
test_sentence_packs = json.load(open(data_path + "test.json", encoding='UTF-8'))

print(len(train_sentence_packs))
token_vocab = Vocab.load_vocab(data_path + 'vocab_tok.vocab')  # token嵌入
post_vocab = Vocab.load_vocab(data_path + 'vocab_post.vocab')  # 位置嵌入
deprel_vocab = Vocab.load_vocab(data_path + 'vocab_deprel.vocab')  # 依存边嵌入
postag_vocab = Vocab.load_vocab(data_path + 'vocab_postag.vocab')  # 词性嵌入
pospair_vocab = Vocab.load_vocab(data_path + 'vocab_pospair.vocab')  # 词性组合嵌入
vocab = (token_vocab, post_vocab, deprel_vocab, postag_vocab, pospair_vocab)

tok_size = len(token_vocab)
post_size = len(post_vocab)  # 相对位置
dep_size = len(deprel_vocab)  # 依存关系索引
postag_size = len(postag_vocab)  # 词性标注索引
pospair_size = len(pospair_vocab)
print(
    "token_vocab: {},post_vocab: {}, deprel_vocab: {}, postag_vocab: {}, pospair_vocab: {}".format(
        len(token_vocab), len(post_vocab), len(deprel_vocab), len(postag_vocab), len(pospair_vocab)
    )
)
# print(
#     "token_vocab: {},post_vocab: {}, deprel_vocab: {}, postag_vocab: {}".format(
#         len(token_vocab), len(post_vocab), len(deprel_vocab), len(postag_vocab)
#     )
# )



def get_spans_label(tags):
    '''for BIO tag'''
    tags = tags.strip().split()
    # print(tags)
    length = len(tags)
    spans = []
    start = -1
    for i in range(length):
        if tags[i].endswith('B'):
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[i].endswith('O'):
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans

def preprocess(data, vocab, use):
    SW_index, MW_index = 0, 0
    token_vocab, post_vocab, deprel_vocab, postag_vocab,pospair_vocab= vocab
    processed = []
    all_data = []
    drop_index = []
    save_index = []
    span_indices = []
    for d in data:
        #分别提取aspect、opinion位置序号,以及三元组标签,
        spans = []
        span_labels = []
        relations = []
        relation_labels = []

        sentence = d['sentence']
        tokens = sentence.strip().split()  # 分离每个token
        # print(tokens)
        deprel = d['deprel']
        head = d['head']
        postag = list(d['postag'])
        sen_length = len(tokens)
        # print(sentence)
        # print(deprel)
        # print(head)

        # for i in range(len(postag)):
        #     if postag[i] == 'NN' or postag[i] == 'jj':
        #         sum=sum+1

        # print(sen_length)

        for triple in d['triples']:
            aspect = triple['target_tags']
            opinion = triple['opinion_tags']
            aspect_span_label = get_spans_label(aspect)
            opinion_span_label = get_spans_label(opinion)
            triple_label = sentiment2id[triple['sentiment']]

            for i in range(len(opinion_span_label)):
                spans.append(aspect_span_label[0])
                span_labels.append(spans2id['aspect'])
                spans.append(opinion_span_label[i])
                span_labels.append(spans2id['opinion'])
                relation_labels.append(triple_label)
        for i in range(len(relation_labels)):
            relations.append(spans[2 * i] + spans[2 * i + 1])
        # print(aspect_span)
        # print(opinion_span)
        # print(triple_span)
        # print(relations)
        # print(relation_labels)

        token = [token_vocab.stoi.get(t, token_vocab.unk_index) for t in tokens]

        depre = [deprel_vocab.stoi.get(t) for t in deprel]
        posta = [postag_vocab.stoi.get(t) for t in postag]

        adj_matrix, label_matrix = head_to_adj(head, depre, sen_length, self_loop=True)
        postag_adj_matrix, postag_matrix = postag_adj(head, postag, deprel, sen_length, 3, pospair_vocab,
                                                      self_loop=False, symmetry=False, Relation=False)
        # print(postag_adj_matrix)
        if use:
            span_token, _ = Span_create(adj_matrix, 8, symmetry=True)
            postag_span, _ = postag_Span_create(postag_adj_matrix, 8, symmetry=True)
            # print(span_token)
            # print('111')
            # print(postag_span)
            span_indices = span_token
            for i in range(len(postag_span)):
                if postag_span[i] not in span_indices:
                    span_indices.append(postag_span[i])

        else:
            len_arrange = torch.arange(0, sen_length)
            max_window = min(sen_length, 8)
            for window in range(1, max_window + 1):
                if window == 1:
                    indics = [(x.item(), x.item()) for x in len_arrange]
                else:
                    res = len_arrange.unfold(0, window, 1)
                    indics = [(idx[0].item(), idx[-1].item()) for idx in res]
                span_indices.extend(indics)

        processed +=[span_indices]
    return processed

    # print(a)
use = True

processed1 = preprocess(train_sentence_packs, vocab, use)
processed2 = preprocess(dev_sentence_packs, vocab, use)
processed3 = preprocess(test_sentence_packs, vocab, use)
end = time.perf_counter()

endtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(endtime)
runTime = (end - start)
print("time：", runTime)
# print(processed)

