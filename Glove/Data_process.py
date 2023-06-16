# -*- coding: utf-8 -*-
# @Time : 2023/2/20 10:49
# @Author : Pan Li
import linecache
import torch
import numpy as np

spans2id = {'invalid': 0, 'aspect': 1, 'opinion': 2}
sentiment2id = {'invalid': 0, 'negative': 1, 'neutral': 2, 'positive': 3}


def load_pretrained_embedding(glove_dir, word_list, dimension_size=300, encoding="utf-8"):
    pre_words = []
    count = 0
    with open(glove_dir + "/glove_words.txt", "r", encoding=encoding) as fopen:
        for line in fopen:
            pre_words.append(line.strip())
    word2offset = {w: i for i, w in enumerate(pre_words)}

    word_vectors = []
    for word in word_list:
        if word in word2offset:
            line = linecache.getline(glove_dir + "/glove.840B.300d.txt", word2offset[word] + 1)
            assert word == line[: line.find(" ")].strip()
            word_vectors.append(
                np.fromstring(line[line.find(" "):].strip(), sep=" ", dtype=np.float32)
            )
            count += 1
        else:
            # init zero
            word_vectors.append(np.zeros(dimension_size, dtype=np.float32))
    print("Loading {}/{} words from vocab...".format(count, len(word_list)))

    return word_vectors

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
                spans.append([start, i])
            start = i
        elif tags[i].endswith('O'):
            if start != -1:
                spans.append([start, i])
                start = -1
    if start != -1:
        spans.append([start, length])
    return spans

class ABSADataLoader(object):
    def __init__(self, dataset, batch_size, vocab, args, shuffle=True):
        self.bacth_size = batch_size
        self.args = args
        self.vocab = vocab

        data = self.preprocess(dataset, vocab, args)
        # print(data[0:20])
        if shuffle:
            indices = np.arange(len(data))
            np.random.shuffle(indices)
            data = [data[idx] for idx in indices]
        self.num_examples = len(data)
        data = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created".format(len(data)))

    def preprocess(self, data, vocab, args):
        token_vocab, post_vocab, deprel_vocab, postag_vocab = vocab
        processed = []
        for d in data:

            spans = []
            span_labels = []
            relations = []
            relation_labels = []

            sentence = d['sentence']
            tokens = sentence.strip().split()  # 分离每个token
            deprel = d['deprel']
            head = d['head']
            postag = list(d['postag'])
            # print(postag)
            sen_length = len(tokens)
            # print(sentence)
            # print(deprel)
            # print(head)

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
            # print(spans)
            # print(span_labels)
            # print(relations)
            # print(relation_labels)
            # print(sentence)
            # print(postag)
            tok = [token_vocab.stoi.get(t, token_vocab.unk_index) for t in tokens]

            deprel = [deprel_vocab.stoi.get(t) for t in deprel]
            postag = [postag_vocab.stoi.get(t) for t in postag]
            # print(tok)
            # print(postag)
            # postag_special = ['NN', 'JJ', 'NNS', 'NNP', 'RB']
            postag_special = ['NN', 'JJ']
            postag_special = [postag_vocab.stoi.get(t) for t in postag_special]
            # print(postag_special)

            # # postion:
            # post = []
            # for i in range(len(spans)):
            #     post_1 = ([j - spans[i][0] for j in range(spans[i][0])]
            #               + [0 for _ in range(spans[i][0], spans[i][1])]
            #               + [j - spans[i][1] + 1 for j in range(spans[i][1], sen_length)])
            #     post_1 = [post_vocab.stoi.get(t, post_vocab.unk_index) for t in post_1]
            #
            #     post.append(post_1)
            processed += [(tok, postag, head, deprel, spans, span_labels, relations, relation_labels, postag_special, sen_length)]
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

        # torch.Size([8, 15]) tok

        # torch.Size([24, 15]) post
        # torch.Size([24, 2])  spans
        # torch.Size([24])     spans_labels
        # torch.Size([8])      count
        # torch.Size([12, 4])  relations
        # torch.Size([12])     relation_labels
        # torch.Size([8])      length
        tok = get_long_tensor(batch[0], batch_size)
        postag = get_long_tensor(batch[1], batch_size)

        head = get_long_tensor(batch[2], batch_size)
        deprel = get_long_tensor(batch[3], batch_size)

        # post = get_expand_tensor(batch[4])

        spans = get_span_tensor(batch[4])

        span_labels = get_span_tensor(batch[5])

        count = Count(batch[5])

        relations = get_relation_span_tensor(batch[6])

        relation_labels = get_relation_span_tensor(batch[7])

        postag_special = torch.LongTensor(batch[8])

        length = torch.LongTensor(batch[9])
        # print(tok)
        # print(tok.size())
        # print(post)
        # print(post.size())
        # print(spans)
        # print(spans.size())
        # print(span_labels)
        # print(span_labels.size())
        # print(count)
        # print(count.size())
        # print(relations)
        # print(relations.size())
        # print(relation_labels)
        # print(relation_labels.size())

        return (tok, postag, head, deprel, spans, span_labels, count, relations, relation_labels, postag_special, length)

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

def get_expand_tensor(tokens_list):
    max_len = 0
    b = []
    for i in range(len(tokens_list)):
        for j in range(len(tokens_list[i])):
            if max_len < len(tokens_list[i][j]):
                max_len = len(tokens_list[i][j])
    for i in range(len(tokens_list)):
        for j in range(len(tokens_list[i])):
            tokens = torch.LongTensor(max_len).fill_(0)
            tokens[:len(tokens_list[i][j])] = torch.LongTensor(tokens_list[i][j])
            b.append(tokens)
    return torch.cat(b).reshape(-1, max_len)

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
def Count(tokens_list):
    count = []
    for i in range(len(tokens_list)):
        count.append(len(tokens_list[i]))
    return torch.LongTensor(count)

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]