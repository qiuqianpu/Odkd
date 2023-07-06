import torch.nn as nn
import torch
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = self.gc2(x, adj)
        # x = 3 * torch.sigmoid(x)
        # x = 5 * F.log_softmax(x, dim=1)
        return x
        # return F.log_softmax(x, dim=1)

