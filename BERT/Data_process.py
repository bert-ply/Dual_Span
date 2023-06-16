# -*- coding: utf-8 -*-
# @Time : 2023/3/1 9:54
# @Author : Pan Li
import torch
import numpy as np
from Common.Tree import Span_create

spans2id = {'invalid': 0, 'aspect': 1, 'opinion': 2}
sentiment2id = {'invalid': 0, 'negative': 1, 'neutral': 2, 'positive': 3}

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

class ABSADataLoader(object):
    def __init__(self, dataset, tokenizer, batch_size, vocab, args, shuffle=True):
        self.bacth_size = batch_size
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab

        data = self.preprocess(dataset, vocab, args)
        if shuffle:
            indices = np.arange(len(data))
            np.random.shuffle(indices)
            data = [data[idx] for idx in indices]
        self.num_examples = len(data)
        data = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created".format(len(data)))

    def fetch_offset(self, pre_text, curr_text):
        start_offset = len(self.tokenizer.encode(pre_text)[:-1])  # 去掉102
        curr_len = len(self.tokenizer.encode(curr_text)[1:-1])  # 去掉101、102
        end_offset = start_offset + curr_len
        return start_offset, end_offset
    def preprocess(self, data, vocab, args):
        token_vocab, post_vocab, deprel_vocab, postag_vocab = vocab
        processed = []
        for d in data:
            spans = []
            span_labels = []
            relations = []
            relation_labels = []
            all_span = []


            sentence = d['sentence']
            tokens = sentence.strip().split()  # 分离每个token
            deprel = d['deprel']
            head = d['head']
            postag = list(d['postag'])
            deprel = [deprel_vocab.stoi.get(t) for t in deprel]
            postag = [postag_vocab.stoi.get(t) for t in postag]
            postag_special = ['NN', 'JJ']
            postag_special = [postag_vocab.stoi.get(t) for t in postag_special]

            inputs = self.tokenizer.encode_plus(sentence, max_length=args.max_seq_len, padding='max_length', truncation=True)
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            token_type_ids = inputs.token_type_ids
            sen_length = len(tokens)
            # print(input_ids)
            # print(attention_mask)
            # print(token_type_ids)
            # print(deprel)
            # print(head)
            # print(postag)
            # print(sen_length)

            for triple in d['triples']:
                aspect = triple['target_tags']
                opinion = triple['opinion_tags']
                aspect_span_label = get_spans_label(aspect)
                opinion_span_label = get_spans_label(opinion)
                triple_label = sentiment2id[triple['sentiment']]
                for i in range(len(opinion_span_label)):
                    a1, a2 = aspect_span_label[0]
                    o1, o2 = opinion_span_label[i]
                    # fetch offsets
                    a_start_idx, a_end_idx = self.fetch_offset(" ".join(tokens[:a1]), " ".join(tokens[a1:a2]))
                    o_start_idx, o_end_idx = self.fetch_offset(" ".join(tokens[:o1]), " ".join(tokens[o1:o2]))
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

            #根据依存树筛选所有可能的span和它们之间的关系
            span_token, label_lst = Span_create(head, deprel, sen_length, args.max_seq_len, args.span_maximum_length)
            for i in range(len(span_token)):
                s1, s2 = span_token[i]
                s1_start_idx, s2_end_idx = self.fetch_offset(" ".join(tokens[:s1]), " ".join(tokens[s1:s2]))
                all_span.append([s1_start_idx, s2_end_idx])

            processed += [(postag, head, deprel, input_ids, attention_mask, token_type_ids, spans, span_labels, relations, relation_labels, all_span, label_lst, postag_special, sen_length)]
        return processed
    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError

        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # print(batch_size)
        # print(batch[0])
        postag = get_long_tensor(batch[0], batch_size)
        head = get_long_tensor(batch[1], batch_size)
        deprel = get_long_tensor(batch[2], batch_size)
        input_ids = get_long_tensor(batch[3], batch_size)
        attention_mask = get_long_tensor(batch[4], batch_size)
        token_type_ids = get_long_tensor(batch[5], batch_size)
        spans = get_span_tensor(batch[6])
        span_labels = get_span_tensor(batch[7])
        relations = get_relation_span_tensor(batch[8])
        relation_labels = get_relation_span_tensor(batch[9])
        span_token = get_span_tensor(batch[10])
        label_lst = torch.LongTensor(batch[11])
        postag_special = torch.LongTensor(batch[12])
        sen_length = torch.LongTensor(batch[13])
        # print(postag)
        # print(postag.size())
        # print(input_ids)
        # print(input_ids.size())
        # print(spans)
        # print(spans.size())
        # print(span_labels)
        # print(span_labels.size())
        # print(relations)
        # print(relations.size())
        # print(relation_labels)
        # print(relation_labels.size())
        # print(all_span)
        # print(all_span.size())
        # print(label_lst)
        # print(label_lst.size())
        # print(sen_length)

        return (postag, head, deprel, input_ids, attention_mask, token_type_ids, spans, span_labels, relations, relation_labels, span_token, label_lst, postag_special, sen_length)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, : len(s)] = torch.LongTensor(s)
    return tokens
def get_span_tensor(tokens_list):
    span = []
    for i in range(len(tokens_list)):
        span.extend(tokens_list[i])
    return torch.LongTensor(span)

def get_relation_span_tensor(tokens_list):
    relation_span = []
    for i in range(len(tokens_list)):
        relation_span.extend(tokens_list[i])
    return torch.LongTensor(relation_span)
def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]
