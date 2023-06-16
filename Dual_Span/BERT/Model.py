# -*- coding: utf-8 -*-
# @Time : 2023/3/1 9:54
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
from transformers import BertModel

class SynBertmodel(nn.Module):
    def __init__(self, args, tokenizer):
        super(SynBertmodel, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.use_span_dep = args.sapn_use_dep
        self.use_triplet_dep = args.triplet_use_dep
        #是否加postag嵌入待定
        self.postag_emb = nn.Embedding(args.postag_size, args.postag_dim, padding_idx=0) if args.postag_dim > 0 else None  # POS emb

        self.bert = BertModel.from_pretrained(args.bert_model)
        encoding_dim = self.bert.config.hidden_size

        # 获取span表示,对span表示进行分类
        self.SpanRepresentation = SpanRepresentation(args)
        if self.use_span_dep:
            span_dim = encoding_dim * 2 + args.span_width_dim + args.dep_dim
        else:
            span_dim = encoding_dim * 2 + args.span_width_dim
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

    def forward(self, input):
        """
        :param x: B * L * D
        :param adj: B * L * L
        :return:
        """
        postag, head, deprel, input_ids, attention_mask, token_type_ids, _, span_labels, relations, relation_labels, span_token, label_lst, postag_special, sen_length = input

        batch_size, sequence_len = input_ids.size()
        batch_max_seq_len = max(sen_length)
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)
        x = bert_output.last_hidden_state
        
        spans, span_indices, label_lst = self.SpanRepresentation(input, x, batch_max_seq_len)
        spans_probability = self.span_fnn(spans)
        
        nz = int(batch_max_seq_len * self.args.span_pruned_threshold)

        target_indices, opinion_indices = self.pruned_target_opinion(spans_probability, nz)#[batchsize,nz]

        candidates, candidate_indices, relation_indices = self.target_opinion_pair_representation(
            spans, span_indices, label_lst, target_indices, opinion_indices)
        candidate_probability = self.pairs_fnn(candidates)

        return spans_probability, span_indices, candidate_probability, candidate_indices
class SpanRepresentation(nn.Module):
    def __init__(self, args):
        super(SpanRepresentation, self).__init__()
        self.span_maximum_length = args.span_maximum_length
        self.span_use_dep = args.sapn_use_dep
        self.use_dep_span = args.use_dep_span
        self.dep_emb = nn.Embedding(args.dep_size, args.dep_dim, padding_idx=0) if args.dep_dim > 0 else None
        self.bucket_bins = [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64]
        self.span_width_embedding = nn.Embedding(len(self.bucket_bins), args.span_width_dim)

    def bucket_embedding(self, width):
        em = [ix for ix, v in enumerate(self.bucket_bins) if width >= v][-1]
        return self.span_width_embedding(torch.LongTensor([em]).cuda())

    def forward(self, input, x, batch_max_seq_len):
        postag, head, deprel, input_ids, attention_mask, token_type_ids, spans, span_labels, relations, relation_labels, span_token, label_lst, postag_special, sen_length = input
        batch_size, _, _ = x.size()

        len_arrange = torch.arange(0, batch_max_seq_len).cuda()
        span_indices = []
        max_window = min(batch_max_seq_len, self.span_maximum_length)
        if self.use_dep_span:
            span_indices = span_token.cpu().tolist()
            label_lst = torch.squeeze(label_lst, dim=0).cpu().tolist()
            indics = []
            for i in range(len(postag[0])):
                if postag[0][i] == 2 or postag[0][i] == 3:
                    if i - 3 < 0 and i + 4 > len(postag[0]):
                        len_arrange = torch.arange(0, len(postag[0])).tolist()
                    elif i - 3 < 0:
                        len_arrange = torch.arange(0, i + 4).tolist()
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

        else:
            for window in range(1, max_window + 1):
                if window == 1:
                    indics = [(x.item(), x.item() + 1) for x in len_arrange]
                else:
                    res = len_arrange.unfold(0, window, 1)
                    indics = [(idx[0].item(), idx[-1].item() + 1) for idx in res]
                span_indices.extend(indics)

        spans = [torch.cat(
            (x[:, s[0], :], x[:, s[1] - 1, :],
             self.bucket_embedding(abs(s[1] - s[0])).repeat(
                 (batch_size, 1)).cuda()),
            dim=1) for s in span_indices]


        # print(span_token)
        # print(label_lst)

        if self.span_use_dep:
            dep_embs = self.dep_emb(torch.tensor(label_lst).cuda())
            # print(dep_embs.size())
            # print(dep_embs)
            SpanR = torch.cat((torch.stack(spans, dim=1), torch.unsqueeze(dep_embs,dim=0)),dim=2)
            return SpanR, span_indices, label_lst

        else:
            return (torch.stack(spans, dim=1)), span_indices, label_lst

class PrunedTargetOpinion:
    #根据预测的得分,设置阈值去筛选所有可能的对，减少计算成本
    def __init__(self):
        pass

    def __call__(self, spans_probability, nz):
        #torch.topk 返回列表最大n个值的位置（从0开始） target 返回预测概率为1（aspect）最大的nz个索引
        target_indices = torch.topk(spans_probability[:, :, spans2id['aspect']], nz, dim=-1).indices
        opinion_indices = torch.topk(spans_probability[:, :, spans2id['opinion']], nz, dim=-1).indices
        return target_indices, opinion_indices

class PairRepresentation(nn.Module):
    def __init__(self, args):
        super(PairRepresentation, self).__init__()
        self.bucket_bins = [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64]
        self.use_dep = args.triplet_use_dep
        self.dep_emb = nn.Embedding(args.dep_size, args.dep_dim, padding_idx=0) if args.dep_dim > 0 else None
        self.distance_embeddings = nn.Embedding(len(self.bucket_bins), args.triplet_distance_dim)

    def min_distance(self, a, b, c, d):
        return min(abs(b - c), abs(a - d))

    def bucket_embedding(self, width, device):
        em = [ix for ix, v in enumerate(self.bucket_bins) if width >= v][-1]
        return self.distance_embeddings(torch.LongTensor([em]).to(device))

    def forward(self, spans, span_indices, label_lst, target_indices, opinion_indices):

        batch_size = spans.size(0)
        device = spans.device

        candidate_indices, relation_indices, triplet_R = [], [], []
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
                # print(R)
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
        if self.use_dep:
            for i in range(len(triplet_R)):
                if len(triplet_R[i]) == 1:
                    triplet_dep = self.dep_emb(torch.tensor(triplet_R[i]).cuda())
                    dep_em.append(triplet_dep)
                else:
                    triplet_dep = torch.unsqueeze(
                        self.dep_emb(torch.tensor(triplet_R[i]).cuda()).sum(dim=0) / len(triplet_R),
                        dim=0).cuda()
                    dep_em.append(triplet_dep)
            dep_embs = torch.cat((dep_em), dim=0).cuda()
            # print(dep_embs.size())
            triplet_SpanR = torch.cat((torch.stack(candidate_pool), torch.unsqueeze(dep_embs, dim=0)), dim=2)
            return triplet_SpanR, candidate_indices, relation_indices
        else:
            return torch.stack(candidate_pool), candidate_indices, relation_indices



