import jsonpickle
from networkx import has_path, shortest_path_length, shortest_path
from networkx.readwrite import json_graph
from src.utils_v2 import *
import yaml
import networkx as nx

def create_store_nx_graph(checkpoint_file, graph_file, device):
    train_df, test_df, val_df, best_model = load_data_model(checkpoint_file, device)
    edge_index, edge_weight = df_to_graph(train_df, True)
    edges, _ = edge_index.split(int(len(edge_weight) / 2), dim=1)
    edges = edges.t().tolist()
    g = nx.Graph(edges)
    with open(graph_file, 'w+') as _file:
        _file.write(jsonpickle.encode(json_graph.adjacency_data(g)))
        print(f'nx graph is stored at: {graph_file}')
    n_users = train_df['user_id_idx'].nunique()
    n_items = train_df['item_id_idx'].nunique()
    return n_users, n_items, train_df, val_df, test_df, edge_index, edge_weight, g

def load_nx_graph(graph_file):
    call_graph = None
    with open(graph_file, 'r+') as _file:
        call_graph = json_graph.adjacency_graph(
            jsonpickle.decode(_file.read()),
            directed=False
        )
    return call_graph

def process_reco_df(csv_path, n_users):
    def index_items(items, n_users):
        return list(map(lambda x: x + n_users, items))
    rec_df = pd.read_csv(csv_path)
    hit_df = rec_df.loc[rec_df.overlap_item.apply(lambda x: len(x) > 2)]
    hit_df = hit_df[['user_id_idx', 'item_id_idx_list', 'top_rlvnt_itm', 'overlap_item']]
    hit_df['item_id_idx_list'] = hit_df['item_id_idx_list'].apply(lambda x: list(map(int, x[1:len(x) - 1].split(', '))))
    hit_df['top_rlvnt_itm'] = hit_df['top_rlvnt_itm'].apply(lambda x: list(map(int, x[1:len(x) - 1].split(', '))))
    hit_df['overlap_item'] = hit_df['overlap_item'].apply(lambda x: list(map(int, x[1:len(x) - 1].split(', '))))
    hit_df.item_id_idx_list = hit_df.item_id_idx_list.apply(lambda x: index_items(x, n_users))
    hit_df.top_rlvnt_itm = hit_df.top_rlvnt_itm.apply(lambda x: index_items(x, n_users))
    hit_df.overlap_item = hit_df.overlap_item.apply(lambda x: index_items(x, n_users))
    return hit_df

def compute_shortest_path(hit_df, graph):
    def path_len(g, s, ds):
        result = []
        for d in ds:
            if has_path(g, s, d):
                result.append(shortest_path_length(g, s, d))
            else:
                result.append(-1)
        return result
    hit_df['shortest_path'] = [path_len(graph, s, ds) for s, ds in zip(hit_df.user_id_idx, hit_df.top_rlvnt_itm)]
    hit_df = hit_df.sort_values(by=['shortest_path'], ascending=False)
    return hit_df

def compute_any5(hit_df):
    def any_5(x):
        for i in x:
            if i > 3:
                return True
        return False
    hit_df['any5'] = hit_df.shortest_path.apply(lambda x: any_5(x))
    return hit_df

def compute_path(hit_df, graph, hit_df_path):
    def hops(user, item_list, any5, graph):
        result = list()
        if any5:
            for i in item_list:
                result.append(shortest_path(graph, user, i))
        return result

    hit_df['paths'] = [hops(a,b,c,graph) for a,b,c in zip(hit_df.user_id_idx, hit_df.top_rlvnt_itm, hit_df.any5)]
    hit_df.to_csv(hit_df_path)
    print(f'Hit_df is stored at: {hit_df_path}')
    return hit_df


def main():
    with open("config.yaml") as config_file:
        config = yaml.safe_load(config_file)
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = config['training']['checkpoints_dir'] + '2023-02-18_071308/'
    recommend_dir = config['inference']['recommendation']+'2023-02-18_071308/'

    n_users, n_items, train_df, val_df, test_df, edge_index, edge_weight, graph = \
        create_store_nx_graph(checkpoint_dir, recommend_dir+'nx_graph.json', device)
    # graph = load_nx_graph(recommend_dir+'nx_graph.json')

    # string to list of int, uniquely identified items
    hit_df = process_reco_df(recommend_dir+'K20-2023-02-18_071308.csv', n_users)
    hit_df = compute_shortest_path(hit_df, graph)
    hit_df = compute_any5(hit_df)
    hit_df = compute_path(hit_df, graph, recommend_dir+'hit_df.csv')


# def glance_df(df):
#     n_users, n_items, glance, _, _ = relabelling(df)
#     glance.item_id_idx = glance.item_id_idx + n_users       # not for train_df
#     edge_index, edge_weight = df_to_graph(glance, True)
#     print(f'n_users {n_users}, n_items {n_items}, len(glance) {len(glance)}, len(edge_weight) {len(edge_weight)}')
#     return n_users, n_items, edge_index, edge_weight
#
# def analysis_df(train_df, test_df, val_df):
#     n_users = train_df.user_id_idx.nunique()
#     n_items = train_df.item_id_idx.nunique()
#     train = train_df
#     train.item_id_idx = train_df.item_id_idx - n_users
#     combined = pd.concat([train, test_df, val_df], ignore_index=True)
#     combined.item_id_idx = combined.item_id_idx + n_users
#     edge_index, edge_weight = df_to_graph(combined, True)
#     print(f'n_users {n_users}, n_items {n_items}, len(combined) {len(combined)}, len(edge_weight) {len(edge_weight)}')
#     return n_users, n_items, edge_index, edge_weight
#
# def create_dgl_graph(n_users, n_items, edge_index, edge_weight):
#     edges = (edge_index[0], edge_index[1])
#     g = dgl.graph(edges)
#     g.edata['weight'] = edge_weight
#     g.ndata['U0-i1-label'] = torch.cat((torch.zeros(n_users), torch.ones(n_items)))
#     return g
#
# def create_writer(writerName):
#     path = '/Users/yingkang/4thBrain/GNN-eCommerce/'+writerName
#     if not os.path.exists(path):
#         writer = Writer(writerName)
#     else:
#         shutil.rmtree(path)
#         writer = Writer(writerName)
#     return writer
#
# def add_graph(writer, graphName, dgl_graph, num_nlabels):
#     writer.add_graph(name=graphName, graph=dgl_graph,
#                      nlabels=dgl_graph.ndata['U0-i1-label'],
#                      num_nlabel_types=num_nlabels,
#                      eweights={'weight': dgl_graph.edata['weight']})
#     writer.close()
#     return writer
# # def add_nlabels():
#     # ground truth label(U0-i1-label)
# def add_subgraph(writer, graphName, subgraphName, dgl_subgraph, nId):
#     writer.add_subgraph(graph_name=graphName, subgraph_name=subgraphName,
#                         node_id=nId,
#                         subgraph_nids=dgl_subgraph.ndata[dgl.NID],
#                         subgraph_eids=dgl_subgraph.edata[dgl.EID])
#     writer.close()
#     return writer


