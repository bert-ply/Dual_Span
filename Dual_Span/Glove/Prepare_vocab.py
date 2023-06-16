# -*- coding: utf-8 -*-
# @Time : 2023/2/17 16:27
# @Author : Pan Li
import json
import argparse
from collections import Counter
from Vocab import Vocab

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare vocab for relation extraction.")
    parser.add_argument("--data_dir", default='../Data/D2/lap14', help="input directory.")
    parser.add_argument("--vocab_dir", default='../Data/D2/lap14', help="Output vocab directory.")
    parser.add_argument("--lower", default=True, help="If specified, lowercase all words.")
    args = parser.parse_args()
    return args

def load_tokens(filename):
    with open(filename, encoding='UTF-8') as infile:
        data = json.load(infile)
        tokens = []
        pos = []
        dep = []
        pospair = []
        max_len = 0
        for d in data:
            sentence = d['sentence'].split()
            tokens.extend(sentence)
            pos.extend(d["postag"])
            dep.extend(d["deprel"])
            n = len(d['postag'])
            tmp_pos = []
            for i in range(n):
                for j in range(n):
                    tup = tuple(sorted([d['postag'][i], d['postag'][j]]))
                    tmp_pos.append(tup)
            pospair.extend(tmp_pos)
            max_len = max(len(sentence), max_len)
    print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
    return tokens, pos, pospair, dep, max_len


def main():
    args = parse_args()
    # input files
    train_file = args.data_dir + "/train.json"
    dev_file = args.data_dir + '/dev.json'
    test_file = args.data_dir + "/test.json"

    # output files
    # token
    vocab_tok_file = args.vocab_dir + "/vocab_tok.vocab"
    # pos_tag
    vocab_pos_file = args.vocab_dir + '/vocab_postag.vocab'
    # pos_tag
    vocab_pospair_file = args.vocab_dir + '/vocab_pospair.vocab'
    # position
    vocab_post_file = args.vocab_dir + "/vocab_post.vocab"
    # dep_rel
    vocab_dep_file = args.vocab_dir + '/vocab_deprel.vocab'

    # load files
    print("loading files...")
    train_tokens, train_pos, train_pospair, train_dep, train_max_len = load_tokens(train_file)
    dev_tokens, dev_pos, dev_pospair, dev_dep, dev_max_len = load_tokens(dev_file)
    test_tokens, test_pos, test_pospair, test_dep, test_max_len = load_tokens(test_file)
    # print(train_tokens)
    # print(train_pos)
    # print(train_dep)
    # print(train_max_len)
    if args.lower:
        train_tokens, dev_tokens, test_tokens = [[t.lower() for t in tokens] for tokens in \
                                                 (train_tokens, dev_tokens, test_tokens)]

    # counters
    token_counter = Counter(train_tokens + dev_tokens + test_tokens)
    print(token_counter)
    pos_counter = Counter(train_pos + dev_pos + test_pos)
    pospair_counter = Counter(train_pospair + dev_pospair + test_pospair)
    dep_counter = Counter(train_dep + dev_dep + test_dep)
    print(pos_counter)
    # print(pospair_counter)
    max_len = max(train_max_len, dev_max_len, test_max_len)
    post_counter = Counter(list(range(-max_len, max_len)))
    # print(post_counter)

    # specials = ["<pad>", "<unk>"]
    # itos = list(specials)
    # print(token_counter)
    # print('************')
    # words_and_frequencies = sorted(token_counter.items(), key=lambda tup: tup[0])
    # print(words_and_frequencies)
    # print('111111111111111')
    # words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
    # print(words_and_frequencies)
    # for word, _ in words_and_frequencies:
    #     itos.append(word)
    # print(itos)
    # stoi = {tok: i for i, tok in enumerate(itos)}
    # print(stoi)

    # build vocab
    print("building vocab...")
    token_vocab = Vocab(token_counter, specials=["<pad>", "<unk>"])
    # print(token_vocab)
    pos_vocab = Vocab(pos_counter, specials=["<pad>", "<unk>"])
    pospair_vocab = Vocab(pospair_counter, specials=["<pad>", "<unk>"])

    dep_vocab = Vocab(dep_counter, specials=["<pad>", "<unk>", "<self>"])
    # print(dep_vocab)
    post_vocab = Vocab(post_counter, specials=["<pad>", "<unk>"])
    print(
        "token_vocab: {}, pos_vocab: {}, pospair_vocab: {}, dep_vocab: {}, post_vocab: {}".format(
            len(token_vocab), len(pos_vocab), len(pospair_vocab), len(dep_vocab), len(post_vocab)
        )
    )
    print("dumping to files...")
    token_vocab.save_vocab(vocab_tok_file)
    pos_vocab.save_vocab(vocab_pos_file)
    pospair_vocab.save_vocab(vocab_pospair_file)
    dep_vocab.save_vocab(vocab_dep_file)
    post_vocab.save_vocab(vocab_post_file)
    print("all done.")



if __name__ == "__main__":
    main()
