# -*- coding: utf-8 -*-
# @Time : 2023/2/21 15:32
# @Author : Pan Li
import torch
import numpy as np
import pandas as pd


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

def postag_adj(head, postag, deprel, length, window, pospair_vocab, self_loop=True, symmetry=True, Relation=False):
    postag_adj_matrix = np.zeros((length,length), dtype=np.float32)
    postag_matrix = np.zeros((length, length), dtype=np.int64)
    for i in range(length):
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
                for k in range(i, max+1):
                    postag_adj_matrix[j][k] = 1
                    postag_matrix[j][k] = pospair_vocab.stoi.get(tuple(sorted([postag[j], postag[k]])))
    if self_loop:
        for i in range(length):
            postag_adj_matrix[i][i] = 1
            postag_matrix[i][i] = pospair_vocab.stoi.get(tuple(sorted([postag[i], postag[i]])))
    if Relation:
        for i in range(length):
            if deprel[i] == 'nsubj':
                if deprel[i] == 'nsubj':
                    if i < head[i] - 1:
                        m = i
                        n = head[i] - 1
                    else:
                        m = head[i] - 1
                        n = i
                postag_adj_matrix[m][n] = 1
                postag_matrix[i][i] = pospair_vocab.stoi.get(tuple(sorted([postag[m], postag[n]])))
                if postag[m] != "NN" or postag[n] != "JJ":
                    if m >= 2:
                        o = m-2
                    else:
                        o = 0
                    for j in range(o, m):
                        postag_adj_matrix[j][i] = 1
                        postag_matrix[j][i] = pospair_vocab.stoi.get(tuple(sorted([postag[j], postag[i]])))
                if postag[head[i]-1] != "NN" or postag[head[i]-1] != "JJ":
                    if n >= 2:
                        o = n-2
                    else:
                        o = 0
                    for j in range(head[i]-3, head[i]-1):
                        postag_adj_matrix[j][head[i]-1] = 1
                        postag_matrix[j][head[i]-1] = pospair_vocab.stoi.get(tuple(sorted([postag[j], postag[head[i]-1]])))
    if symmetry:
         for i in range(len(postag)):
             for j in range(i+1, length):
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
                spans.append([i, j])
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
                postag_spans.append([i, j])
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