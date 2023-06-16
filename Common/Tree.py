# -*- coding: utf-8 -*-
# @Time : 2023/2/21 15:32
# @Author : Pan Li
import torch
import numpy as np
import pandas as pd

def Span_create(head, deprel, sen_length, max_seq_len, span_maximum_length):
    spans = []
    labels = []
    adj_i, label_i = head_to_adj(max_seq_len, head, deprel, sen_length, directed=False,
                                 self_loop=True, symmetry=True)
    # print(adj_i)
    # print(label_i)
    # adj = pd.DataFrame(adj_i)
    # label = pd.DataFrame(label_i)
    # adj.to_excel('adj.xlsx',index=False)
    # label.to_excel('label.xlsx', index=False)
    for i in range(sen_length):
        for j in range(sen_length):
            if adj_i[i][j] == 1 and j-i < span_maximum_length:
                spans.append([i, j+1])
            if label_i[i][j] != 0 and j-i < span_maximum_length:
                labels.append(label_i[i][j])
    return spans, labels

def head_to_adj(sent_len, head, label, len_, directed=False, self_loop=True, symmetry=False):
    """
    Convert a sequence of head indexes into a 0/1 matirx and label matrix.
    """
    adj_matrix = np.zeros((sent_len, sent_len), dtype=np.float32)
    label_matrix = np.zeros((sent_len, sent_len), dtype=np.int64)

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
    if symmetry:
        for i in range(len_):
            for j in range(i + 1, len_):
                adj_matrix[j][i] = 0
                label_matrix[j][i] = 0
    return adj_matrix, label_matrix



