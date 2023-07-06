import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))#.to(device)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))#.to(device)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # 旧的forward没有考虑batchsize的影响
    # def forward(self, input, adj):
    #     # print('----------GCN----------')
    #     # print('adj: ', adj.shape)
    #     # print('input: ', input.shape)
    #     # print('weight: ', self.weight.shape)
    #     support = torch.mm(input, self.weight)
    #     output = torch.spmm(adj, support)
    #     # print('output: ', output.shape)
    #     if self.bias is not None:
    #         return output + self.bias
    #     else:
    #         return output

    def forward(self, input, adj):
        # print('----------GCN----------')
        # print('adj: ', adj.shape)
        # print('input: ', input.shape)
        # print('weight: ', self.weight.shape)
        support = torch.matmul(input, self.weight)
        # print(self.weight)
        output = torch.matmul(adj, support)
        # print('output: ', output.shape)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
