
from ipdb.__main__ import set_trace
import sys
import os
from torch.nn import modules
sys.path.remove(os.getcwd())
import clip
print(clip.__file__)
sys.path.insert(0, os.getcwd())
# sys.path.append('')
import decord
import torch
import torch.nn as nn
import yaml
from dotmap import DotMap
from glob import glob
import os.path as osp
from tqdm import tqdm
from torch.autograd import Variable
import math


class CNN(nn.Module):
    def __init__(self, depth, kernel_size, padding, in_planes, out_planes, channel_reduction_ratio=1, is_bn=True, fc_lr5=False, hashead=False):
        super().__init__()
        modules = []
        for i in range(depth):
            modules.append(nn.Conv1d(in_planes, in_planes//channel_reduction_ratio, kernel_size, padding=padding))
            in_planes = in_planes//channel_reduction_ratio
            if is_bn:
                modules.append(nn.BatchNorm1d(in_planes))
            modules.append(nn.ReLU(inplace=True))
        modules.append(nn.AdaptiveMaxPool1d(1))
        if hashead:
            modules.append(nn.Conv1d(in_planes, out_planes, kernel_size, padding=padding))
        self.forwarder = nn.Sequential(*modules)
        self.fc_lr5 = fc_lr5

    def forward(self, input):
        return self.forwarder(input).squeeze(-1)

    def get_optim_policies(self):
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        word_embedding = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                ps = list(m.parameters())
                conv_cnt += 1
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])
            elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                bn_cnt += 1
                # later BN's are frozen
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.Embedding):
                word_embedding.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             },
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
            {'params': word_embedding,'lr_mult': 1, 'decay_mult': 1,
             'name': "word_embedding"}
        ]



class bottleneck_CNN(CNN):
    def __init__(self, depth, kernel_size, padding, in_planes, out_planes, channel_reduction_ratio=1, bottleneck_reduction=4, is_bn=True, fc_lr5=False, hashead=False, firstlayer_reduction=None):
        super(CNN, self).__init__()
        modules = []
        for i in range(depth-1):
            if i==0 and firstlayer_reduction is not None:
                # first layer channel reduction
                modules.append(nn.Conv1d(in_planes, in_planes//firstlayer_reduction, 1, padding=0))
                modules.append(nn.BatchNorm1d(in_planes//firstlayer_reduction))
                modules.append(nn.ReLU(inplace=True))
                in_planes //= firstlayer_reduction

            bottleneck_dim = in_planes//bottleneck_reduction
            if not hashead and i==depth-1-1:
                next_stage_dim = out_planes
            else:
                next_stage_dim = in_planes//channel_reduction_ratio
            # bottleneck reduction
            modules.append(nn.Conv1d(in_planes, bottleneck_dim, 1, padding=0))
            modules.append(nn.BatchNorm1d(bottleneck_dim))
            modules.append(nn.ReLU(inplace=True))
            # Conv
            modules.append(nn.Conv1d(bottleneck_dim, bottleneck_dim, kernel_size, padding=padding))
            if is_bn:
                modules.append(nn.BatchNorm1d(bottleneck_dim))
            modules.append(nn.ReLU(inplace=True))
            # botlleneck inflation
            modules.append(nn.Conv1d(bottleneck_dim, next_stage_dim, 1, padding=0))
            modules.append(nn.BatchNorm1d(next_stage_dim))
            modules.append(nn.ReLU(inplace=True))
            # channel reduction
            in_planes = next_stage_dim

            

        modules.append(nn.AdaptiveMaxPool1d(1))
        if hashead:
            modules.append(nn.Conv1d(in_planes, out_planes, kernel_size, padding=padding))
        self.forwarder = nn.Sequential(*modules)
        self.fc_lr5 = fc_lr5


class PositionalEncoding(nn.Module):
    """
    Positional encoding from the Transformer paper.
    """
    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
                          
    def forward(self, x):
       x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
       return self.dropout(x)


class TemporalCrossTransformer(nn.Module):
    """
    A temporal cross transformer for a single tuple cardinality. E.g. pairs or triples.
    """
    def __init__(self, args, temporal_set_size=3):
        super(TemporalCrossTransformer, self).__init__()
       
        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        self.pe = PositionalEncoding(self.args.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)

        self.k_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()
        self.v_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()

        self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)
        
        self.class_softmax = torch.nn.Softmax(dim=1)
        
        # generate all tuples
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = nn.ParameterList([nn.Parameter(torch.tensor(comb), requires_grad=False) for comb in frame_combinations])
        self.tuples_len = len(self.tuples) 
    
    
    def forward(self, support_set, support_labels, queries):
        n_queries = queries.shape[0]
        n_support = support_set.shape[0]
        
        # static pe
        support_set = self.pe(support_set)
        queries = self.pe(queries)

        # construct new queries and support set made of tuples of images after pe
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]
        support_set = torch.stack(s, dim=-2)
        queries = torch.stack(q, dim=-2)

        # apply linear maps
        support_set_ks = self.k_linear(support_set)
        queries_ks = self.k_linear(queries)
        support_set_vs = self.v_linear(support_set)
        queries_vs = self.v_linear(queries)
        
        # apply norms where necessary
        mh_support_set_ks = self.norm_k(support_set_ks)
        mh_queries_ks = self.norm_k(queries_ks)
        mh_support_set_vs = support_set_vs
        mh_queries_vs = queries_vs
        
        unique_labels = torch.unique(support_labels)

        # init tensor to hold distances between every support tuple and every target tuple
        all_distances_tensor = torch.zeros(n_queries, self.args.way, device=queries.device)

        for label_idx, c in enumerate(unique_labels):
        
            # select keys and values for just this class
            class_k = torch.index_select(mh_support_set_ks, 0, extract_class_indices(support_labels, c))
            class_v = torch.index_select(mh_support_set_vs, 0, extract_class_indices(support_labels, c))
            k_bs = class_k.shape[0]

            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2,-1)) / math.sqrt(self.args.trans_linear_out_dim)

            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0,2,1,3)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1)
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)]
            class_scores = torch.cat(class_scores)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1, self.tuples_len)
            class_scores = class_scores.permute(0,2,1,3)
            
            # get query specific class prototype         
            query_prototype = torch.matmul(class_scores, class_v)
            query_prototype = torch.sum(query_prototype, dim=1)
            
            # calculate distances from queries to query-specific class prototypes
            diff = mh_queries_vs - query_prototype
            norm_sq = torch.norm(diff, dim=[-2,-1])**2
            distance = torch.div(norm_sq, self.tuples_len)
            
            # multiply by -1 to get logits
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor[:,c_idx] = distance
        
        return_dict = {'logits': all_distances_tensor}
        
        return return_dict


if __name__=='__main__':
    model = bottleneck_CNN(depth=3, kernel_size=3, padding=1, in_planes=25000, out_planes=10, channel_reduction_ratio=1, bottleneck_reduction=4, hashead=True, firstlayer_reduction=8)
    set_trace()