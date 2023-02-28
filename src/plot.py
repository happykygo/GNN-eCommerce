import os
import shutil
import dgl
from gnnlens import Writer
from matplotlib import pyplot as plt
import pandas as pd
import torch
import networkx as nx
from src.utils_v2 import df_to_graph, relabelling

def create_dgl_graph(df):
    n_users, n_items, glance, _, _ = relabelling(df)
    glance.item_id_idx = glance.item_id_idx + n_users  # not for train_df
    edge_index, edge_weight = df_to_graph(glance, True)
    print(f'n_users {n_users}, n_items {n_items}, len(glance) {len(glance)}, len(edge_weight) {len(edge_weight)}')
    edges = (edge_index[0], edge_index[1])
    g = dgl.graph(edges)
    g.edata['weight'] = edge_weight
    g.ndata['U0-i1-label'] = torch.cat((torch.ones(n_users), torch.zeros(n_items)))
    return g

def add_gnnlens_graph(graphName, dgl_graph, num_nlabels):
    path = '/Users/yingkang/4thBrain/GNN-eCommerce/' + graphName
    if not os.path.exists(path):
        writer = Writer(graphName)
    else:
        shutil.rmtree(path)
        writer = Writer(graphName)

    writer.add_graph(name=graphName, graph=dgl_graph,
                     nlabels=dgl_graph.ndata['U0-i1-label'],
                     num_nlabel_types=num_nlabels,
                     eweights={'weight': dgl_graph.edata['weight']})
    writer.close()
    return writer

def nx_graph_df(hit_df, user_id):

    def check(df, im):
        y = pd.DataFrame()
        y.paths = df.paths.apply(lambda x: list(map(int, x.split(', '))))
        total = set()
        for i in y.paths:
            total.update(i)
        assert len(total) == (im.u.nunique()+im.i.nunique())

    source = hit_df.loc[hit_df.user_id_idx == user_id]
    path = list(source.paths)[0]
    # for path in source.paths:
    df = pd.DataFrame(path[2:len(path) - 2].split('], ['), columns=(['paths']))
    df[['u1', 'i1', 'u2', 'i2', 'u3', 'i3']] = df.paths.str.split(", ", expand=True)
    im = df[['u1', 'i1']].rename(columns={'u1': 'u', 'i1': 'i'})
    temp = df[['u2', 'i1']].rename(columns={'u2': 'u', 'i1': 'i'})
    im = pd.concat((im, temp)).drop_duplicates()
    temp = df[['u2', 'i2']].rename(columns={'u2': 'u', 'i2': 'i'})
    im = pd.concat((im, temp)).drop_duplicates()
    temp = df[['u3', 'i2']].rename(columns={'u3': 'u', 'i2': 'i'})
    im = pd.concat((im, temp)).drop_duplicates()
    temp = df[['u3', 'i3']].rename(columns={'u3': 'u', 'i3': 'i'})
    im = pd.concat((im, temp)).drop_duplicates()
    im = im.loc[~im.u.isna()]

    check(df, im)
    return im

def create_nx_graph(im_df, hit_df, start_node):
    edges1 = [list([int(a), int(b)]) for a, b in zip(im_df.u, im_df.i)]
    user_nodes = list(map(int, list(im_df.u.unique())))
    item_nodes = list(map(int, list(im_df.i.unique())))
    user_nodes.remove(start_node)

    G = nx.Graph()
    G.add_node(start_node, bipartite=2)  # target user
    G.add_nodes_from(user_nodes, bipartite=0)   # users
    G.add_nodes_from(item_nodes, bipartite=1)   # items
    G.add_edges_from(edges1)

    color_dict = {0: 'tab:orange', 1: 'tab:blue', 2: 'tab:orange'}
    color_list = [color_dict[i[1]] for i in G.nodes.data('bipartite')]
    size_dict = {0: 500, 1: 500, 2: 1000}
    size_list = [size_dict[i[1]] for i in G.nodes.data('bipartite')]

    pos = nx.spring_layout(G)

    df = hit_df.loc[hit_df.user_id_idx == start_node]
    top = list(df.top_rlvnt_itm)[0]
    top = set(map(int, top[1:len(top) - 1].split(', ')))

    plt.figure(1, figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_size=1300, node_color='tab:red')
    nx.draw_networkx_nodes(G, pos, nodelist=top, node_size=800, node_color='tab:red')
    nx.draw(G, pos=pos, with_labels=True, node_color=color_list, font_size=10, node_size=size_list)
    plt.show()