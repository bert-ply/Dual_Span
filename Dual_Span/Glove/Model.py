# -*- coding: utf-8 -*-
# @Time : 2023/2/20 22:11
# @Author : Pan Li
import torch
import torch.nn
import torch.nn as nn

import itertools
from torch.autograd import Variable
from torch.nn import init
import numpy as np
from Common.Tree import Span_create
from Data_process import spans2id

class Synastemodel(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        self.in_dim = args.emb_dim + args.postag_dim
        self.args = args
        self.use_span_dep = args.sapn_use_dep
        self.use_triplet_dep = args.triplet_use_dep
        self.emb = nn.Embedding(args.tok_size, args.emb_dim, padding_idx=0)
        if emb_matrix is not None:
            self.emb.weight = nn.Parameter(emb_matrix.cuda(), requires_grad=False)
        self.postag_emb = nn.Embedding(args.postag_size, args.postag_dim, padding_idx=0) if args.postag_dim > 0 else None  # POS emb
        # self.post_emb = nn.Embedding(args.post_size, args.post_dim, padding_idx=0) if args.post_dim > 0 else None       # position emb

        #LSTM
        input_size = self.in_dim
        self.rnn = nn.LSTM(input_size, args.rnn_hidden, args.rnn_layers, batch_first=True, \
                           dropout=args.rnn_dropout, bidirectional=args.bidirect)
        if args.bidirect:
            self.indim = args.rnn_hidden * 2
        else:
            self.indim = args.rnn_hidden

        self.rnn_drop = nn.Dropout(args.rnn_dropout)
        self.in_drop = nn.Dropout(args.input_dropout)

        #获取span表示,对span表示进行分类
        self.span_representation = SpanRepresentation(args)
        if self.use_span_dep:
            span_dim = self.indim * 2 + args.span_width_dim + args.dep_dim
        else:
            span_dim = self.indim * 2 + args.span_width_dim
        self.span_fnn = torch.nn.Sequential(
            nn.Linear(span_dim, args.fnn_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(args.fnn_hidden_dim, args.fnn_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(args.fnn_hidden_dim, args.class_num, bias=True),
            nn.Softmax(-1)
        )
        # 剪枝策略,返回候选池概率较大的nz个
        self.pruned_target_opinion = PrunedTargetOpinion()
        # 返回aspect-opinion pair 表示并对其进行分类
        self.target_opinion_pair_representation = PairRepresentation(args)
        if self.use_triplet_dep:
            pairs_dim = 2 * span_dim + args.triplet_distance_dim + args.dep_dim
        else:
            pairs_dim = 2 * span_dim + args.triplet_distance_dim
        self.pairs_fnn = torch.nn.Sequential(
            nn.Linear(pairs_dim, args.fnn_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(args.fnn_hidden_dim, args.fnn_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(args.fnn_hidden_dim, args.sentiment_class, bias=True),
            nn.Softmax(-1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.span_fnn.named_parameters():
            if "weight" in name:
                init.xavier_normal_(param)
        for name, param in self.pairs_fnn.named_parameters():
            if "weight" in name:
                init.xavier_normal_(param)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(
            batch_size, self.args.rnn_hidden, self.args.rnn_layers, self.args.bidirect
        )
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, input):
        """
        :param x: B * L * D
        :param adj: B * L * L
        :return:
        """
        # torch.Size([8, 15]) tok

        # torch.Size([24, 15]) post
        # torch.Size([24, 2])  spans
        # torch.Size([24])     spans_labels
        # torch.Size([8])      count
        # torch.Size([12, 4])  relations
        # torch.Size([12])     relation_labels
        # torch.Size([8])      length
        # tok, postag, head, deprel, spans, span_labels, count, relations, relation_labels, postag_special, length
        tok, postag, head, deprel, _, span_labels, count, relations, relation_labels, postag_special, length = input
        # tok = []
        # pos = []
        # seq_len = []
        # for i in range(len(count)):
        #     tok.append(token[i].repeat(count[i], 1))
        #     pos.append(postag[i].repeat(count[i], 1))
        #     seq_len.append(length[i].repeat(count[i]))
        # tok = torch.cat(tok).cuda()
        # pos = torch.cat(pos).cuda()
        # seq_len = torch.cat(seq_len).cuda()

        word_embs = self.emb(tok)
        embs = [word_embs]
        if self.args.postag_dim > 0:
            embs += [self.postag_emb(postag)]
        embs = torch.cat(embs, dim=2)   #[batch_num, seq_len, sum-2-dim]
        embs = self.in_drop(embs)
        # RNN-BiLSTM
        sent_output = self.rnn_drop(
            self.encode_with_rnn(embs, length.cpu(), tok.size()[0])
        )  # [B*, seq_len, H] [batch_num, seq_len, rnn_hidden]
        # print(sent_output.size())
        batch_max_seq_len = max(length)
        spans, span_indices, label_lst = self.span_representation(input, sent_output, batch_max_seq_len)
        # print(spans.size())

        spans_probability = self.span_fnn(spans)#[batchsize, span_number, 3]

        nz = int(batch_max_seq_len * self.args.span_pruned_threshold)

        target_indices, opinion_indices = self.pruned_target_opinion(spans_probability, nz)#[batchsize,nz]

        # spans[batch_size,span_number,1230], span_indices [span_number,2], target_indices[batchsize,nz], opinion_indices[batchsize,nz]
        #candidates[batch_size,nz*nz,600*4+30*2+pair_distance_dim],candidate_indices[2,nz*nz,4]
        candidates, candidate_indices, relation_indices = self.target_opinion_pair_representation(
            spans, span_indices, label_lst, target_indices, opinion_indices)
        # print(candidates.size())
        # candidates[batch_size,nz*nz, 4]
        candidate_probability = self.pairs_fnn(candidates)
        # print(candidate_probability.size())
        # print(candidate_indices)
        # batch_len = sent_output.size(0)
        # # batch span indices
        # span_indices = [span_indices for _ in range(batch_len)]

        return spans_probability, span_indices, candidate_probability, candidate_indices

class SpanRepresentation(nn.Module):
    def __init__(self, args):
        super(SpanRepresentation, self).__init__()
        self.span_maximum_length = args.span_maximum_length
        self.use_dep_span = args.use_dep_span
        self.sapn_use_dep = args.sapn_use_dep
        self.dep_emb = nn.Embedding(args.dep_size, args.dep_dim, padding_idx=0) if args.dep_dim > 0 else None
        self.bucket_bins = [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64]
        self.span_width_embedding = nn.Embedding(len(self.bucket_bins), args.span_width_dim)

    def bucket_embedding(self, width):
        em = [ix for ix, v in enumerate(self.bucket_bins) if width >= v][-1]
        return self.span_width_embedding(torch.LongTensor([em]).cuda())

    def forward(self, input, x, batch_max_seq_len):

        tok, postag, head, deprel, spans, span_labels, count, relations, relation_labels, postag_special, length = input

        # print(tok)
        # print(postag)
        # print(len(postag[0]))
        # print(postag.size())
        # print(postag_special)
        # print(tok)
        # print(tok.size())
        # print(deprel)
        # print(deprel.size())
        # print(length)
        # print(length.size())
        # print(batch_max_seq_len)
        #x [batch, len ,rnn_hidden]
        batch_size, sequence_length, _ = x.size()
        
        len_arrange = torch.arange(0, batch_max_seq_len).cuda()
        span_indices = []
        max_window = min(batch_max_seq_len, self.span_maximum_length)
        
        span_token, label_lst = [], []
        for i in range(len(length)):
            span_i, label_i = Span_create(head[i], deprel[i], length[i], batch_max_seq_len, self.span_maximum_length)
            span_token.extend(span_i)
            label_lst.extend(label_i)
        if self.use_dep_span:
            span_indices = span_token
            # print(span_indices)
            # print(label_lst)
            # print('----')
            indics = []
            for i in range(len(postag[0])):
                if postag[0][i] == 2 or postag[0][i] == 3:
                    if i - 3 < 0 and i + 4 > len(postag[0]):
                        len_arrange = torch.arange(0, len(postag[0])).tolist()
                    elif i - 3 < 0:
                        len_arrange = torch.arange(0, i+4).tolist()
                    elif i + 4 > len(postag[0]):
                        len_arrange = torch.arange(i - 3, len(postag[0])).tolist()
                    else:
                        len_arrange = torch.arange(i - 3, i + 4).tolist()
                    for index1 in len_arrange:
                        for index2 in len_arrange:
                            if index1 < index2 and index1 <= i and index2 >= i and [index1, index2 + 1] not in indics:
                                indics.append([index1, index2 + 1])
            for i in range(len(indics)):
                if indics[i] not in span_indices:
                    span_indices.append(indics[i])
                    label_lst.append(0)
            # print(span_indices)
        else:
            for window in range(1, max_window + 1):
                if window == 1:
                    indics = [(x.item(), x.item()+1) for x in len_arrange]
                else:
                    res = len_arrange.unfold(0, window, 1)
                    indics = [(idx[0].item(), idx[-1].item()+1) for idx in res]
                span_indices.extend(indics)
        # print(span_token)
        # print(x.size())
        spans = [torch.cat(
            (x[:, s[0], :], x[:, s[1]-1, :],
             self.bucket_embedding(abs(s[1] - s[0])).repeat(
                 (batch_size, 1)).cuda()),
            dim=1) for s in span_indices]
        # print(torch.stack(spans, dim=1))
        # print(torch.stack(spans, dim=1).size())
        if self.sapn_use_dep:
            dep_embs = self.dep_emb(torch.tensor(label_lst).cuda())
            # print(dep_embs.size())
            # print(dep_embs)
            SpanR = torch.cat((torch.stack(spans, dim=1), torch.unsqueeze(dep_embs,dim=0)),dim=2)
            return SpanR, span_indices, label_lst

        else:
            return (torch.stack(spans, dim=1)), span_indices, label_lst

class PairRepresentation(nn.Module):
    def __init__(self, args):
        super(PairRepresentation, self).__init__()
        self.bucket_bins = [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64]
        self.triplet_use_dep = args.triplet_use_dep
        self.dep_emb = nn.Embedding(args.dep_size, args.dep_dim, padding_idx=0) if args.dep_dim > 0 else None
        self.distance_embeddings = nn.Embedding(len(self.bucket_bins), args.triplet_distance_dim)

    def min_distance(self, a, b, c, d):
        return min(abs(b - c), abs(a - d))

    def bucket_embedding(self, width, device):
        em = [ix for ix, v in enumerate(self.bucket_bins) if width >= v][-1]
        return self.distance_embeddings(torch.LongTensor([em]).to(device))

    # spans[batch_size, span_number,1230], span_indices [span_number,2], target_indices[batchsize,nz], opinion_indices[batchsize,nz]
    def forward(self, spans, span_indices, label_lst, target_indices, opinion_indices):
        """
       :param spans:
       :param span_indices:
       :param target_indices:
       :type
       :param opinion_indices:
       :return:
           candidate_indices :
               List[List[Tuple(a,b,c,d)]]
           relation_indices :
               List[List[Tuple(span1,span2)]]
       """
        batch_size = spans.size(0)
        device = spans.device
        # candidate_indices :[(a,b,c,d)]
        # relation_indices :[(span1,span2)]
        candidate_indices, relation_indices, triplet_R = [], [], []
        # print(span_indices)
        # print(label_lst)
        # print(target_indices)
        # print(opinion_indices)
        for batch in range(batch_size):
            pairs = list(itertools.product(target_indices[batch].cpu().tolist(), opinion_indices[batch].cpu().tolist())) #opinion和aspect最大的进行匹配，循环迭代
            relation_indices.append(pairs)
            candidate_ind = []
            for pair in pairs:
                a, b = span_indices[pair[0]]
                c, d = span_indices[pair[1]]
                candidate_ind.append((a, b, c, d))
                #计算跨度对之间的关系triplet_R
                triplet_R_l, triplet_R_r = [], []
                if c>=b:
                    for j in range(a, b):
                        triplet_R_l.append(j)
                    for k in range(c + 1, d + 1):
                        triplet_R_r.append(k)
                else:
                    for j in range(c, d):
                        triplet_R_l.append(j)
                    for k in range(a + 1, b + 1):
                        triplet_R_r.append(k)
                R = list(itertools.product(triplet_R_l, triplet_R_r))
                flag = 0
                x = []
                for idx in range(len(R)):
                    for id, indices in enumerate(span_indices):
                        if list(R[idx]) == indices:
                            x.append(label_lst[id])
                            flag = 1
                if flag == 0:
                    triplet_R.append([0])
                else:
                    triplet_R.append(x)

            candidate_indices.extend(candidate_ind)
        # print(candidate_indices)
        # print(triplet_R)
        # print('-----')
        # print(relation_indices)
        candidate_pool = []
        for batch in range(batch_size):
            relations = [
                torch.cat((spans[batch, c[0], :], spans[batch, c[1], :],
                           self.bucket_embedding(
                               self.min_distance(*span_indices[c[0]], *span_indices[c[1]]), device).squeeze(0))
                          , dim=0) for c in relation_indices[batch]]
            candidate_pool.append(torch.stack(relations))
        # print(torch.stack(candidate_pool).size())
        dep_em = []
        if self.triplet_use_dep:
            for i in range(len(triplet_R)):
                if len(triplet_R[i]) == 1:
                    triplet_dep = self.dep_emb(torch.tensor(triplet_R[i]).cuda())
                    dep_em.append(triplet_dep)
                else:
                    triplet_dep = torch.unsqueeze(self.dep_emb(torch.tensor(triplet_R[i]).cuda()).sum(dim=0) / len(triplet_R),
                                                  dim=0).cuda()
                    dep_em.append(triplet_dep)
            dep_embs = torch.cat((dep_em), dim=0).cuda()
            # print(dep_embs.size())
            triplet_SpanR = torch.cat((torch.stack(candidate_pool), torch.unsqueeze(dep_embs, dim=0)), dim=2)
            return triplet_SpanR, candidate_indices, relation_indices
        else:
            return torch.stack(candidate_pool), candidate_indices, relation_indices


class PrunedTargetOpinion:
    #根据预测的得分,设置阈值去筛选所有可能的对，减少计算成本
    def __init__(self):
        pass

    def __call__(self, spans_probability, nz):
        #torch.topk 返回列表最大n个值的位置（从0开始） target 返回预测概率为1（aspect）最大的nz个索引
        target_indices = torch.topk(spans_probability[:, :, spans2id['aspect']], nz, dim=-1).indices
        opinion_indices = torch.topk(spans_probability[:, :, spans2id['opinion']], nz, dim=-1).indices
        return target_indices, opinion_indices

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()