# -*- coding: utf-8 -*-
# @Time : 2023/2/27 10:50
# @Author : Pan Li
import numpy as np
import torch
import itertools
from transformers import BertModel, BertTokenizer, BertConfig
import json
from Vocab import Vocab
import pandas as pd

sentiment2id = {'invalid': 0, 'negative': 1, 'neutral': 2, 'positive': 3}

train_sentence_packs = json.load(open('./Data/res14/test.json', encoding='UTF-8'))

my_bert_model_path = './My_BERT_Model'

tokenizer = BertTokenizer.from_pretrained(my_bert_model_path)

print(train_sentence_packs[0:5])
token_vocab = Vocab.load_vocab('./Data/res14/vocab_tok.vocab')  # 位置嵌入
post_vocab = Vocab.load_vocab('./Data/res14/vocab_post.vocab')  # 位置嵌入
deprel_vocab = Vocab.load_vocab('./Data/res14/vocab_deprel.vocab')  # 依存边嵌入
pospair_vocab = Vocab.load_vocab('./Data/res14/vocab_pospair.vocab')
postag_vocab = Vocab.load_vocab('./Data/res14/vocab_postag.vocab')  # 词性嵌入
vocab = (token_vocab, post_vocab, deprel_vocab, postag_vocab, pospair_vocab)
tok_size = len(token_vocab)
post_size = len(post_vocab)  # 相对位置
deprel_size = len(deprel_vocab)  # 依存关系索引
postag_size = len(postag_vocab)  # 词性标注索引
pospair_size = len(pospair_vocab)
print(
    "token_vocab: {},post_vocab: {}, deprel_vocab: {}, postag_vocab: {}, pospair_vocab: {}".format(
        len(token_vocab), len(post_vocab), len(deprel_vocab), len(postag_vocab), len(pospair_vocab)
    )
)

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
def head_to_adj(head, label, len_, directed=False, self_loop=True):
    """
    Convert a sequence of head indexes into a 0/1 matirx and label matrix.
    """
    adj_matrix = np.zeros((len_, len_), dtype=np.float32)
    label_matrix = np.zeros((len_, len_), dtype=np.int64)

    head = head[:len_]
    label = label[:len_]
    # print('tokens', tokens)
    # print('head', head, len(head))
    # print('label', label)
    for idx, head in enumerate(head):
        # if idx in asp_idx:
        #     for k in asp_idx:
        #         adj_matrix[idx][k] = 1
        #         label_matrix[idx][k] = 2
        if head != 0:
            adj_matrix[idx, head - 1] = 1
            label_matrix[idx, head - 1] = label[idx]
        else:
            if self_loop:
                adj_matrix[idx, idx] = 1
                label_matrix[idx, idx] = 2
                continue
        if not directed:
            adj_matrix[head - 1, idx] = 1
            label_matrix[head - 1, idx] = label[idx]
        if self_loop:
            adj_matrix[idx, idx] = 1
            label_matrix[idx, idx] = 2
    return adj_matrix, label_matrix
def postag_adj_1(sent_len, postag, posta, pospair_vocab, postag_special, token_range, len_, window, self_loop=True,
               Relation=True):
    """
    Convert a sequence of head indexes into a 0/1 matirx and label matrix.
    """
    postag_adj_matrix = np.zeros((sent_len, sent_len), dtype=np.float32)
    postag_matrix = np.zeros((sent_len, sent_len), dtype=np.int64)

    # assert not isinstance(postag, list)
    # postag = postag[:len_].tolist()
    a = np.array(token_range)
    for i in range(len_):
        a[i] = [token_range[i][0] - 1, token_range[i][1] - 2]
    # print(a)
    for i in range(len(postag)):
        if posta[i] == postag_special[0] or posta[i] == postag_special[1]:
            if i - window <= 0 and i + window + 1 >= len(postag):
                min = 0
                max = len_
            elif i - window <= 0:
                min = 0
                max = i + window + 1
            elif i + window + 1 >= len(postag):
                min = i - window
                max = len_
            else:
                min = i - window
                max = i + window + 1
            for j in range(min, max):
                start = a[j][0]
                end = a[j][1]
                for k in range(min, max):
                    s, e = a[k][0], a[k][1]
                    for row in range(start, end + 1):
                        for col in range(s, e + 1):
                            postag_adj_matrix[row][col] = 1
                            postag_matrix[row][col] = pospair_vocab.stoi.get(tuple(sorted([postag[j], postag[k]])))
    if Relation:
        start = []
        end = []
        for i in range(len_):
            if posta[i] == postag_special[0]:
                start.append(i)
            if posta[i] == postag_special[1]:
                end.append(i)
        for j in range(len(start)):
            for k in range(len(end)):
                postag_adj_matrix[start[j]][end[k]] = 1
                postag_adj_matrix[end[k]][start[j]] = 1
                postag_matrix[start[j]][end[k]] = pospair_vocab.stoi.get(
                    tuple(sorted([postag[start[j]], postag[end[k]]])))
                postag_matrix[end[k]][start[j]] = pospair_vocab.stoi.get(
                    tuple(sorted([postag[end[k]], postag[start[j]]])))
    if self_loop:
        for i in range(len_):
            start = a[i][0]
            end = a[i][1]
            for row in range(start, end + 1):
                postag_adj_matrix[row][row] = 1
                postag_matrix[row][row] = pospair_vocab.stoi.get(tuple(sorted([postag[i], postag[i]])))
    return postag_adj_matrix, postag_matrix


def postag_adj(sent_len, head, postag, deprel, token_range, length, window, pospair_vocab, self_loop=True, symmetry=True, Relation=True):
    postag_adj_matrix = np.zeros((sent_len,sent_len), dtype=np.float32)
    postag_matrix = np.zeros((sent_len, sent_len), dtype=np.int64)

    range1 = np.array(token_range)
    for i in range(length):
        range1[i] = [token_range[i][0] - 1, token_range[i][1] - 2]
    print(range1)
    for i in range(length):
        # ["DT", "NNP", "NNS", "VBP", "RB", "RB", "VBN", "DT", "NN", "NN", "."]
        if postag[i] == 'NN' or postag[i] == 'JJ':
            # if i - window <= 0 and i + window >= length-1:
            #     min = 0
            #     max = length-1
            # elif i - window <= 0:
            #     min = 0
            #     max = i + window
            # elif i + window >= length-1:
            #     min = i - window
            #     max = length-1
            # else:
            #     min = i - window
            #     max = i + window
            min, max = windows(i, window, length)
            for j in range(min, i+1):
                start, end = range1[j][0], range1[j][1]
                for k in range(i, max+1):
                    s, e = range1[k][0], range1[k][1]
                    for row in range(start, end+1):
                        for col in range(s, e + 1):
                            postag_adj_matrix[row][col] = 1
                            postag_matrix[row][col] = pospair_vocab.stoi.get(tuple(sorted([postag[j], postag[k]])))
    if self_loop:
        for i in range(length):
            start, end = range1[i][0], range1[i][1]
            for row in range(start, end + 1):
                postag_adj_matrix[row][row] = 1
                postag_matrix[row][row] = pospair_vocab.stoi.get(tuple(sorted([postag[i], postag[i]])))
    if Relation:
        for i in range(length):
            # deprel = ["det", "compound", "nsubj", "aux", "advmod", "advmod", "root", "det", "compound", "obj", "punct"]
            # head = [3, 3, 7, 7, 7, 7, 0, 10, 10, 7, 7]
            if deprel[i] == 'nsubj':
                if i < head[i] - 1:
                    m = i
                    n = head[i] - 1
                else:
                    m = head[i] - 1
                    n = i
                start, end = range1[m][0], range1[m][1]
                s, e = range1[n][0], range1[n][1]
                for row in range(start, end + 1):
                    for col in range(s, e + 1):
                        postag_adj_matrix[row][col] = 1
                        postag_matrix[row][col] = pospair_vocab.stoi.get(tuple(sorted([postag[m], postag[n]])))
                if postag[m] != "NN" or postag[m] != "JJ":
                    s, e = range1[m][0], range1[m][1]
                    if m >= 2:
                        o = m-2
                    else:
                        o = 0
                    for j in range(o, m):
                        start, end = range1[j][0], range1[j][1]
                        for row in range(start, e):
                            postag_adj_matrix[row][e] = 1
                            postag_matrix[row][e] = pospair_vocab.stoi.get(tuple(sorted([postag[m], postag[n]])))
                if postag[n] != "NN" or postag[n] != "JJ":
                    s, e = range1[n][0], range1[n][1]
                    if n >= 2:
                        o = n-2
                    else:
                        o = 0
                    for j in range(o, n):
                        start, end = range1[j][0], range1[j][1]
                        for row in range(start, e):
                            postag_adj_matrix[row][e] = 1
                            postag_matrix[row][e] = pospair_vocab.stoi.get(tuple(sorted([postag[m], postag[n]])))
    if symmetry:
         for i in range(sent_len):
             for j in range(i+1, sent_len):
                 if postag_adj_matrix[i][j] != 0:
                     postag_adj_matrix[j][i] = postag_adj_matrix[i][j]
                     postag_matrix[j][i] = postag_matrix[i][j]
    return postag_adj_matrix, postag_matrix


def Span_create(label_adj, span_maximum_length, symmetry=True):
    spans = []
    labels = []
    label_adj = label_adj.tolist()
    if symmetry:
        for i in range(len(label_adj[0])):
            for j in range(i + 1, len(label_adj[0])):
                label_adj[j][i] = 0
    for i in range(len(label_adj[0])):
        for j in range(len(label_adj[0])):
            if label_adj[i][j] != 0 and j-i < span_maximum_length:
                spans.append([i, j+1])
            if label_adj[i][j] != 0 and j-i < span_maximum_length:
                labels.append(label_adj[i][j])
    return spans, labels

def postag_Span_create(posta_adj, span_maximum_length, symmetry=True):
    postag_spans = []
    postag_labels = []
    posta_adj = posta_adj.tolist()
    if symmetry:
        for i in range(len(posta_adj[0])):
            for j in range(i + 1, len(posta_adj[0])):
                posta_adj[j][i] = 0
    for i in range(len(posta_adj[0])):
        for j in range(len(posta_adj[0])):
            if posta_adj[i][j] != 0 and j-i < span_maximum_length:
                postag_spans.append([i, j+1])
            if posta_adj[i][j] != 0 and j-i < span_maximum_length:
                postag_labels.append(posta_adj[i][j])
    return postag_spans, postag_labels

def windows(i, window, length):
    if i - window <= 0 and i + window >= length - 1:
        min = 0
        max = length - 1
    elif i - window <= 0:
        min = 0
        max = i + window
    elif i + window >= length - 1:
        min = i - window
        max = length - 1
    else:
        min = i - window
        max = i + window
    return min, max


# def fetch_offset(pre_text, curr_text):
#     start_offset = len(tokenizer.encode(pre_text)[:-1])  # 去掉102
#     curr_len = len(tokenizer.encode(curr_text)[1:-1])  # 去掉101、102
#     end_offset = start_offset + curr_len
#     return start_offset, end_offset
# def offset_mapping(tokens, sent_len):
#     offset_mapping = []
#     len_arrange = torch.arange(0, sent_len)
#     indic = [(x.item(), x.item() + 1) for x in len_arrange]
#     for i in range(len(indic)):
#         s1, s2 = indic[i]
#         s1_start_idx, s2_end_idx = fetch_offset(" ".join(tokens[:s1]), " ".join(tokens[s1:s2]))
#         offset_mapping.append([s1_start_idx, s2_end_idx])
#     return offset_mapping


# sentence = "The Apple engineers have not yet discovered the delete key ."
#
# tokens = sentence.strip().split()
# print(len(tokens))
# inputs = tokenizer.encode_plus('delete', max_length=128, padding='max_length', truncation=True)
# print(inputs)
# input_ids = [101, 100, 100, 6145, 2031, 2025, 2664, 3603, 1996, 100, 3145, 1012, 102]
# print(inputs.input_ids)
seq_len = 14

postag = ["DT", "NNP", "NNS", "VBP", "RB", "RB", "VBN", "DT", "NN", "NN", "."]
deprel = ["det", "compound", "nsubj", "aux", "advmod", "advmod", "root", "det", "compound", "obj", "punct"]
head = [3, 3, 7, 7, 7, 7, 0, 10, 10, 7, 7]
token_range = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 11], [11, 12], [12, 13]]


depre = [deprel_vocab.stoi.get(t) for t in deprel]
# posta = [postag_vocab.stoi.get(t) for t in postag]
# postag_special = ['NN', 'JJ']
# postag_special = [postag_vocab.stoi.get(t) for t in postag_special]
# postag_adj_matrix, postag_matrix = postag_adj_1(seq_len - 2, postag, posta, pospair_vocab, postag_special, token_range,
#                                               10, 3)
# postag_adj_matrix, postag_matrix = postag_adj(seq_len-2, head, postag, deprel, token_range, len(head), 3, pospair_vocab)
# print(postag_adj_matrix)
# print(postag_matrix)
# span_1, label = postag_Span_create(postag_matrix, 8)
# print(span_1)
#
# adj_matrix, label_matrix = head_to_adj(head, depre, len(head), self_loop=True)
# span_token, _ = Span_create(label_matrix, 8)
# print(span_token)


def preprocess(data, vocab):
    token_vocab, post_vocab, deprel_vocab, postag_vocab, pospair_size = vocab
    all_data = []
    drop_index = []
    save_index = []
    index = 0
    for d in data:
        index += 1
        #分别提取aspect、opinion位置序号,以及三元组标签,
        aspect_span = []
        opinion_span = []
        triple_span = []
        all_span = []
        sentence = d['sentence']
        tokens = sentence.strip().split()  # 分离每个token
        # print(tokens)
        deprel = d['deprel']
        head = d['head']
        postag = list(d['postag'])
        sen_length = len(tokens)
        print(sentence)
        print(deprel)
        print(head)
        print(postag)
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
                aspect_span.extend(aspect_span_label)
                triple_span.append(triple_label)
            opinion_span.extend(opinion_span_label)
            all_span.extend(aspect_span_label)
            all_span.extend(opinion_span_label)
        # print(aspect_span)
        # print(opinion_span)
        # print(triple_span)
        print(all_span)
        # tok = [token_vocab.stoi.get(t, token_vocab.unk_index) for t in tokens]
        depre = [deprel_vocab.stoi.get(t) for t in deprel]

        posta = [postag_vocab.stoi.get(t) for t in postag]
        # print(postag_special)
        # postag_adj_matrix, postag_matrix = postag_adj(head, postag, deprel, sen_length, 3, pospair_vocab)
        # span_1, label = postag_Span_create(postag_matrix, 8)
        # # print(label)
        # adj_matrix, label_matrix = head_to_adj(head, depre, sen_length, self_loop=True)
        # span_token, _ = Span_create(label_matrix, 8)
        # for i in range(len(span_1)):
        #     if span_1[i] not in span_token:
        #         span_token.append(span_1[i])
        # print(span_token)
        #
        # tab = 0
        # for i in range(len(all_span)):
        #     for j in range(len(span_token)):
        #         if all_span[i] == span_token[j]:
        #             tab +=1
        # if tab != len(all_span):
        #     print('no-------------------------------------------------------------------------------------------------------')
        #     print(index)
        #     drop_index.append(index-1)
        # else:
        #     print(index)
        #     save_index.append(index-1)

    return drop_index,save_index

    # print(a)
drop_index, save_index = preprocess(train_sentence_packs,vocab)
print(drop_index)
print(len(drop_index))
print(len(train_sentence_packs))

data = [train_sentence_packs[i] for i in save_index]
print(len(data))

print(int(10*0.05))

a= torch.randn(1,3,4)
print(a)
print(a[:,0,:])
print(a[:,1,:])