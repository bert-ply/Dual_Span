# -*- coding: utf-8 -*-
# @Time : 2023/2/24 17:17
# @Author : Pan Li

import torch

def gold_labels(span_indices, spans, span_labels):
    """
    Organizing gold labels and indices
    :param span_indices:
    :param spans:
    :param span_labels:
    :return:
        gold_indices:
        gold_labels:
    """
    # gold span labels
    gold_labels = [0]*len(span_indices)
    for i in range(len(spans)):
        # print(spans[i])
        for batch_idx, indices in enumerate(span_indices):
            if spans[i].cpu().tolist() == list(indices):
                gold_labels[batch_idx] = span_labels[i].cpu().tolist()
    return span_indices, gold_labels
