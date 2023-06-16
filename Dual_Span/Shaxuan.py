# -*- coding: utf-8 -*-
# @Time : 2022/12/28 16:40
# @Author : Pan Li
import math
import torch
import numpy as np
from Common.Tree import head_to_adj,Span_create
import json
from Vocab import Vocab
import pandas as pd

spans2id = {'invalid': 0, 'aspect': 1, 'opinion': 2}
sentiment2id = {'invalid': 0, 'negative': 1, 'neutral': 2, 'positive': 3}

train_sentence_packs = json.load(open('./Data/D2/res15/test.json', encoding='UTF-8'))



print(len(train_sentence_packs))
token_vocab = Vocab.load_vocab('./Data/D2/res15/vocab_tok.vocab')  # 位置嵌入
post_vocab = Vocab.load_vocab('./Data/D2/res15/vocab_post.vocab')  # 位置嵌入
deprel_vocab = Vocab.load_vocab('./Data/D2/res15/vocab_deprel.vocab')  # 依存边嵌入
postag_vocab = Vocab.load_vocab('./Data/D2/res15/vocab_postag.vocab')  # 词性嵌入
vocab = (token_vocab, post_vocab, deprel_vocab, postag_vocab)
tok_size = len(token_vocab)
post_size = len(post_vocab)  # 相对位置
deprel_size = len(deprel_vocab)  # 依存关系索引
postag_size = len(postag_vocab)  # 词性标注索引
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
def Span_create(head,  deprel, sen_length):
    spans = []
    span_token_mask = []
    # adj_i, label_i = head_to_adj(102, head, tokens, deprel, sen_length, directed=False,
    #                              self_loop=True)
    adj_i = head_to_adj(102, head, deprel, sen_length, directed=False,
                                 self_loop=True)
    for i in range(sen_length):
        for j in range(sen_length):
            if adj_i[i][j] == 1:
                spans.append([i, j])
                mask = torch.zeros(sen_length)
                if i == j:
                    mask[j] = 1
                else:
                    mask[i:j + 1] = 1
                span_token_mask.append(mask)
    return spans, span_token_mask
def preprocess(data, vocab):
    SW_index, MW_index = 0, 0
    token_vocab, post_vocab, deprel_vocab, postag_vocab = vocab
    all_data = []
    drop_index = []
    save_index = []
    index = 0
    ina = 0
    sum = 0
    for d in data:
        index += 1
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
        ina = ina+1
        sum = sum+sen_length
        print(postag)
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
            # print(aspect_span_label)
            # print(opinion_span_label)
            # print(triple_label)
            #解决exp：aspect[[3, 5]] opinion[[0, 0], [2, 2], [8, 8]]
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
        for j in range(len(relation_labels)):
            if relations[j][1]-relations[j][0]==0 and relations[j][3]-relations[j][2]==0:
                SW_index = SW_index+1
            else:
                MW_index = MW_index+1

        tok = [token_vocab.stoi.get(t, token_vocab.unk_index) for t in tokens]
        deprel = [deprel_vocab.stoi.get(t) for t in deprel]
        postag = [postag_vocab.stoi.get(t) for t in postag]

        # span_token, span_mask_token = Span_create(head, deprel, sen_length)
        # print(span_token)
        # tab = 0
        # for i in range(len(all_span)):
        #     for j in range(len(span_token)):
        #         if all_span[i] == span_token[j]:
        #             tab +=1
        # if tab != len(all_span):
        #     print('no-------------------------------------------------------------------------------------------------------')
        #     print(index)q
        #     drop_index.append(index-1)
        # else:
        #     print(index)
        #     save_index.append(index-1)

    return SW_index, MW_index, ina, sum

    # print(a)
drop_index, save_index, i ,sum = preprocess(train_sentence_packs,vocab)
print(drop_index)
print(save_index)
print(sum/i)
print(sum)
# print(len(drop_index))
# print(len(train_sentence_packs))

# data = [train_sentence_packs[i] for i in save_index]
# print(len(data))
#
# file = './data/D1/Sample_lap14/dev.json'
# with open(file, 'w') as outfile:
#     json.dump(data, outfile)

