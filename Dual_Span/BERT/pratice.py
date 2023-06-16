# -*- coding: utf-8 -*-
# @Time : 2023/3/2 10:56
# @Author : Pan Li
import math
import torch
import torch.nn as nn
import numpy as np
# from Common.Tree import head_to_adj
from torch.nn import functional as F
import json
from Vocab import Vocab
# from Common.Glod_labels import gold_labels
import itertools
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

sentiment2id = {'invalid': 0, 'negative': 1, 'neutral': 2, 'positive': 3}
spans2id = {'invalid': 0, 'aspect': 1, 'opinion': 2}
train_sentence_packs = json.load(open('../Data/Sample_res15/train.json'))
print(len(train_sentence_packs))

token_vocab = Vocab.load_vocab('../Data/Sample_res15/vocab_tok.vocab')  # 位置嵌入
post_vocab = Vocab.load_vocab('../Data/Sample_res15/vocab_post.vocab')  # 位置嵌入
deprel_vocab = Vocab.load_vocab('../Data/Sample_res15/vocab_deprel.vocab')  # 依存边嵌入
postag_vocab = Vocab.load_vocab('../Data/Sample_res15/vocab_postag.vocab')  # 词性嵌入
vocab = (token_vocab, post_vocab, deprel_vocab, postag_vocab)
tok_size = len(token_vocab)
post_size = len(post_vocab)  # 相对位置
deprel_size = len(deprel_vocab)  # 依存关系索引
postag_size = len(postag_vocab)  # 词性标注索引
print(
    "token_vocab: {},post_vocab: {}, deprel_vocab: {}, postag_vocab: {}".format(
        len(token_vocab), len(post_vocab), len(deprel_vocab), len(postag_vocab)
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
                spans.append([start, i])
                start = -1
    if start != -1:
        spans.append([start, length])
    return spans
def fetch_offset(pre_text, curr_text):
    start_offset = len(tokenizer.encode(pre_text)[:-1])  # 去掉102
    curr_len = len(tokenizer.encode(curr_text)[1:-1])  # 去掉101、102
    end_offset = start_offset + curr_len
    return start_offset, end_offset

def preprocess(data, vocab):
    token_vocab, post_vocab, deprel_vocab, postag_vocab = vocab
    all_data = []
    for d in data:
        #分别提取aspect、opinion位置序号,以及三元组标签,

        spans = []
        span_labels = []
        relations = []
        relation_labels = []

        sentence = d['sentence']
        tokens = sentence.strip().split()  # 分离每个token
        deprel = d['deprel']
        head = d['head']
        postag = list(d['postag'])
        deprel = [deprel_vocab.stoi.get(t) for t in deprel]
        postag = [postag_vocab.stoi.get(t) for t in postag]

        inputs = tokenizer.encode_plus(sentence, max_length=128, padding='max_length', truncation=True)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        token_type_ids = inputs.token_type_ids
        seq_len = len([i for i in input_ids if i != 0])
        # print(input_ids)
        # print(attention_mask)
        # print(token_type_ids)
        # print(deprel)
        # print(head)
        # print(postag)
        # print(sen_length)

        for triple in d['triples']:
            aspect = triple['target_tags']
            # print(aspect)
            opinion = triple['opinion_tags']
            aspect_span_label = get_spans_label(aspect)
            opinion_span_label = get_spans_label(opinion)
            triple_label = sentiment2id[triple['sentiment']]
            # print(aspect_span_label)
            # print(opinion_span_label)
            for i in range(len(opinion_span_label)):
                a1, a2 = aspect_span_label[0]
                o1, o2 = opinion_span_label[i]
                # fetch offsets
                a_start_idx, a_end_idx = fetch_offset(" ".join(tokens[:a1]), " ".join(tokens[a1:a2]))
                o_start_idx, o_end_idx = fetch_offset(" ".join(tokens[:o1]), " ".join(tokens[o1:o2]))
                spans.append([a_start_idx, a_end_idx])
                span_labels.append(spans2id['aspect'])
                spans.append([o_start_idx, o_end_idx])
                span_labels.append(spans2id['opinion'])
                relation_labels.append(triple_label)
        for i in range(len(relation_labels)):
            relations.append(spans[2 * i] + spans[2 * i + 1])
        # print(spans)
        # print(span_labels)
        # print(relations)
        # print(relation_labels)
        # print('---')

        all_data += [(postag, head, deprel, input_ids, attention_mask, token_type_ids, spans, span_labels, relations, relation_labels, seq_len)]
    return all_data

data = preprocess(train_sentence_packs[0:10], vocab)

print(data[0:5])
print(len(data))

# bert编码器排序
# # sent = 'Spreads and toppings are great - though a bit pricey .'
# sent = "The pizza is yummy and I like the atmoshpere ."
# encode_words = tokenizer.convert_ids_to_tokens(tokenizer.encode(sent))
# print(encode_words)
# print(tokenizer.encode_plus(sent))

data = [data[i: i + 8] for i in range(0, len(data), 8)]
batch = data[0]
batch_size = len(batch)
batch = list(zip(*batch))
print(batch_size)

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, : len(s)] = torch.LongTensor(s)
    return tokens
# print(batch[0])
print(batch[3])
print(batch[4])
print(batch[5])
input_ids = get_long_tensor(batch[3], batch_size)
print(input_ids.size())