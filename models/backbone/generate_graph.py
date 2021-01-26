import numpy as np
import dgl
import torch
import torch.nn.functional as F

def graph_creat(uv_dict):
    #print("-------------------------------图结构化建模开始-------------------------------")
    G_src_nodes = []
    G_target_nodes = []
    for u_i in list(uv_dict.keys()):
        src_nodes = list(map(int, list(uv_dict[u_i].keys())))
        #print(src_nodes)
        num_src = len(src_nodes)
        target_nodes = [int(u_i) for i in range(num_src)]
        G_target_nodes.extend(target_nodes)
        G_src_nodes.extend(src_nodes)
    G = dgl.DGLGraph((np.array(G_src_nodes), np.array(G_target_nodes)))
    #print("-------------------------------图结构化建模结束-------------------------------")

    G_edges_num = G.number_of_edges()
    G.edata['rela'] = torch.zeros(G_edges_num, 1)
    #G.ndata['degrees'] = G.out_degrees(G.nodes()).float() + 1.0
    for u_i in list(uv_dict.keys()):
        if (len(uv_dict[u_i].keys()) == 0):
            continue
        rela = torch.tensor(list(uv_dict[u_i].values())).float().unsqueeze(1)
        G.edata['rela'][G.edge_ids(list(map(int, list(uv_dict[u_i].keys()))), int(u_i))] = F.softmax(rela, dim=0)

    return G

