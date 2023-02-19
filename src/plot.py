# import pandas as pd
# import torch
import os
import shutil
from src.utils_v2 import *
import yaml
import dgl
from gnnlens import Writer


def glance_df(df):
    n_users, n_items, glance, _, _ = relabelling(df)
    glance.item_id_idx = glance.item_id_idx + n_users       # not for train_df
    edge_index, edge_weight = df_to_graph(glance, True)
    print(f'n_users {n_users}, n_items {n_items}, len(glance) {len(glance)}, len(edge_weight) {len(edge_weight)}')
    return n_users, n_items, edge_index, edge_weight

def analysis_df(train_df, test_df, val_df):
    n_users = train_df.user_id_idx.nunique()
    n_items = train_df.item_id_idx.nunique()
    train = train_df
    train.item_id_idx = train_df.item_id_idx - n_users
    combined = pd.concat([train, test_df, val_df], ignore_index=True)
    combined.item_id_idx = combined.item_id_idx + n_users
    edge_index, edge_weight = df_to_graph(combined, True)
    print(f'n_users {n_users}, n_items {n_items}, len(combined) {len(combined)}, len(edge_weight) {len(edge_weight)}')
    return n_users, n_items, edge_index, edge_weight

def create_dgl_graph(n_users, n_items, edge_index, edge_weight):
    edges = (edge_index[0], edge_index[1])
    g = dgl.graph(edges)
    g.edata['weight'] = edge_weight
    g.ndata['U0-i1-label'] = torch.cat((torch.zeros(n_users), torch.ones(n_items)))
    return g

def create_writer(writerName):
    path = '/Users/yingkang/4thBrain/GNN-eCommerce/'+writerName
    if not os.path.exists(path):
        writer = Writer(writerName)
    else:
        shutil.rmtree(path)
        writer = Writer(writerName)
    return writer

def add_graph(writer, graphName, dgl_graph, num_nlabels):
    writer.add_graph(name=graphName, graph=dgl_graph,
                     nlabels=dgl_graph.ndata['U0-i1-label'],
                     num_nlabel_types=num_nlabels,
                     eweights={'weight': dgl_graph.edata['weight']})
    writer.close()
    return writer


# def add_nlabels():
    # ground truth label(U0-i1-label)

def add_subgraph(writer, graphName, subgraphName, dgl_subgraph, nId):
    writer.add_subgraph(graph_name=graphName, subgraph_name=subgraphName,
                        node_id=nId,
                        subgraph_nids=dgl_subgraph.ndata[dgl.NID],
                        subgraph_eids=dgl_subgraph.edata[dgl.EID])
    writer.close()
    return writer

def main():
    with open("config.yaml") as config_file:
        config = yaml.safe_load(config_file)
    checkpoint_dir = config['training']['checkpoints_dir']+'2023-02-15_060043/'
    # all saved models can be load
    load_data_model(checkpoint_dir)
