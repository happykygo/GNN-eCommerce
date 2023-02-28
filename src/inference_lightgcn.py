from src.lightgcn import LightGCN
from src.utils_v2 import interact_matrix, df_to_graph, load_data_model, pos_item_list
import pandas as pd
import torch
import networkx as nx
from networkx.readwrite import json_graph
from networkx import has_path, shortest_path_length, shortest_path
import jsonpickle
import yaml
import argparse


class InferenceLightGCN:
    def __init__(self, checkpoint_dir, gpu=0):
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        train_df, test_df, val_df, best_model = load_data_model(checkpoint_dir, device)
        self.edge_index, self.edge_weight = df_to_graph(train_df, True)
        self.edge_index = self.edge_index.to(device)
        self.edge_weight = self.edge_weight.to(device)

        n_users = train_df['user_id_idx'].nunique()
        n_items = train_df['item_id_idx'].nunique()
        train_df['item_id_idx'] = train_df['item_id_idx'] - n_users
        i_m_matrix = interact_matrix(train_df, n_users, n_items)  # keep in cpu

        val_test = pd.concat([test_df, val_df], ignore_index=True)
        val_test = pos_item_list(val_test).sort_values(by=['user_id_idx'])
        user_list = list(val_test['user_id_idx'])
        self.interactions_t = torch.index_select(i_m_matrix, 0, torch.tensor(user_list)).to_dense()

        hyperparams = best_model['hyperparams']
        self.test_model = LightGCN(n_users + n_items, hyperparams['latent_dim'], hyperparams['n_layers'])
        self.test_model.load_state_dict(best_model['model_state_dict'])
        self.test_model.to(device)

        self.user_list = user_list
        self.n_users = n_users
        self.n_items = n_items
        self.val_test = val_test

    def recommendation(self, k):
        topK_df = self.test_model.recommendK(self.edge_index, self.edge_weight, self.n_users, self.n_items,
                                          self.interactions_t, self.user_list, k)
        topK_precision, topK_recall, metrics = self.test_model.MARK_MAPK(self.val_test, topK_df, k)
        # result = pd.merge(self.val_test, topK_df, how='left', left_on='user_id_idx', right_on='user_ID')
        # result['overlap_item'] = [list(set(a).intersection(b)) for a, b in zip(result.item_id_idx_list, result.top_rlvnt_itm)]
        return topK_precision, topK_recall, metrics

    def create_store_nx_graph(self, graph_file):
        edges, _ = self.edge_index.split(int(len(self.edge_weight) / 2), dim=1)
        edges = edges.t().tolist()
        g = nx.Graph(edges)
        with open(graph_file, 'w+') as _file:
            _file.write(jsonpickle.encode(json_graph.adjacency_data(g)))
            print(f'nx graph is stored at: {graph_file}')
        return g

    def load_nx_graph(self, graph_file):
        # graph = load_nx_graph(reco_dir+'nx_graph.json')
        call_graph = None
        with open(graph_file, 'r+') as _file:
            call_graph = json_graph.adjacency_graph(
                jsonpickle.decode(_file.read()),
                directed=False
            )
        return call_graph

    def prepare_hit_df(self, metrics):
        """
        hit_df contains len(metrics.overlap_item) > 0.
        item_id in hit-df need to be uniquely identified
        :param metrics:
        :return:
        """
        def index_items(items, n_users):
            return list(map(lambda x: x + n_users, items))

        hit_df = metrics.loc[metrics.overlap_item.apply(lambda x: len(x) > 0)]
        hit_df.item_id_idx_list = hit_df.item_id_idx_list.apply(lambda x: index_items(x, self.n_users))
        hit_df.top_rlvnt_itm = hit_df.top_rlvnt_itm.apply(lambda x: index_items(x, self.n_users))
        hit_df.overlap_item = hit_df.overlap_item.apply(lambda x: index_items(x, self.n_users))
        return hit_df

    def compute_paths(self, hit_df, graph):
        """
        1. Compute path_len between target user and each recommended items
        2. Check if any path_len > 3
        3. Compute paths from target user to each recommended items
        :param hit_df:
        :param graph:
        :return:
        """
        def path_len(g, s, ds):
            result = []
            for d in ds:
                if has_path(g, s, d):
                    result.append(shortest_path_length(g, s, d))
                else:
                    result.append(-1)
            return result
        def longerThan3(x):
            for i in x:
                if i > 3:
                    return True
            return False
        def paths(user, item_list, graph):
            result = list()
            for i in item_list:
                result.append(shortest_path(graph, user, i))
            return result

        hit_df['path_lens'] = [path_len(graph, s, ds) for s, ds in zip(hit_df.user_id_idx, hit_df.top_rlvnt_itm)]
        hit_df = hit_df.sort_values(by=['path_lens'], ascending=False)

        hit_df['longer_than_3'] = hit_df.shortest_path.apply(lambda x: longerThan3(x))

        hit_df['paths'] = [paths(a, b, graph) for a, b in zip(hit_df.user_id_idx, hit_df.top_rlvnt_itm)]
        return hit_df

def main(gpu, checkpoint):
    with open("config.yaml") as config_file:
        config = yaml.safe_load(config_file)

    checkpoint_dir = config['training']['checkpoints_dir'] + checkpoint
    inference_dir = config['inference']['recommendation'] + checkpoint

    k = 20
    inferenceModel = InferenceLightGCN(checkpoint_dir, gpu)
    topK_precision, topK_recall, metrics = inferenceModel.recommendation(k)
    print(f"Inference Precision@{k}: {topK_precision:>7f}, Recall@{k}: {topK_recall:>7f}")
    metrics.to_csv(inference_dir+'/K'+str(k)+'-' + checkpoint + '.csv')
    print(f'topK.csv is stored.')

    nx_graph = inferenceModel.create_store_nx_graph(inference_dir+'/nx_graph.json')
    # if nx_graph is none:
    #   nx_graph = load_nx_graph(graph_file)

    hit_df = inferenceModel.prepare_hit_df(metrics)
    hit_df = inferenceModel.compute_paths(hit_df, nx_graph)
    hit_df.to_csv(inference_dir+'/hit_df.csv')
    print(f'hit_df.csv is stored.')


if __name__ == "__main__":
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-g", "--gpu", required=True, help="which gpu to use")
    ap.add_argument("-c", "--checkpoint", required=True, help="which checkpoint to use")
    args = vars(ap.parse_args())
    main(args['gpu'], args['checkpoint'])

