import dgl
import torch.nn.functional as F
import dgl.function as fn
import torch.nn as nn
from torch.nn import Parameter
from dgl.nn.pytorch.conv import GraphConv
from utils import *

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCN, self).__init__()
        self.gcn1 = GraphConv(in_feats, in_feats)
        self.gcn2 = GraphConv(in_feats, out_feats)

    def forward(self, g, features):
        h = features
        h = F.relu(self.gcn1(g, h))
        h = F.relu(self.gcn2(g, h))
        g.ndata['h'] = h
        h = dgl.mean_nodes(g, 'h')
        return h