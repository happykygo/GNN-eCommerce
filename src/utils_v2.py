from sklearn import preprocessing as pp
import torch
import random
import pandas as pd
import numpy as np
from datetime import datetime


def purchase_users(df):
    """
    Filter out users who never purchase. Cannot evaluate model performance on them.
    :param df:
    :return:
    """
    u_id_filter = df.loc[df['weight'] == 1.00].user_id.unique()
    df = df.loc[df['user_id'].isin(u_id_filter)]
    return df


def sync_nodes(train_df, val_df, test_df):
    r"""
    Remove user/items in val_df/test_df but not in train_df
    Remove users with NO purchase in val_df/test_df
    :param train_df:
    :param val_df:
    :param test_df:
    :return:
    """
    train_user_ids = train_df['user_id'].unique()
    train_item_ids = train_df['item_id'].unique()
    val_df = val_df[
        (val_df['user_id'].isin(train_user_ids)) & (val_df['item_id'].isin(train_item_ids))]
    test_df = test_df[
        (test_df['user_id'].isin(train_user_ids)) & (test_df['item_id'].isin(train_item_ids))]
    val_df = purchase_users(val_df)
    test_df = purchase_users(test_df)
    return val_df, test_df


def relabelling(train_df, val_df, test_df):
    r"""
    Relabelling user/ item nodes.
    :param train_df:
    :param val_df:
    :param test_df:
    :return:
    """
    le_user = pp.LabelEncoder()
    le_item = pp.LabelEncoder()
    train_df['user_id_idx'] = le_user.fit_transform(train_df['user_id'].values)
    train_df['item_id_idx'] = le_item.fit_transform(train_df['item_id'].values)
    val_df['user_id_idx'] = le_user.transform(val_df['user_id'].values)
    val_df['item_id_idx'] = le_item.transform(val_df['item_id'].values)
    test_df['user_id_idx'] = le_user.transform(test_df['user_id'].values)
    test_df['item_id_idx'] = le_item.transform(test_df['item_id'].values)

    train_df = train_df.drop(columns=['user_id', 'item_id'])
    val_df = val_df.drop(columns=['user_id', 'item_id'])
    test_df = test_df.drop(columns=['user_id', 'item_id'])

    n_users = train_df['user_id_idx'].nunique()
    n_items = train_df['item_id_idx'].nunique()
    return n_users, n_items, train_df, val_df, test_df


def pos_item_list(df):
    r"""
    Generate Positive Item List for df
    :param df:
    :return:
    """
    u_posI_List = df.loc[df['weight'] == 1]
    u_posI_List = u_posI_List.groupby('user_id_idx')['item_id_idx'].apply(list).reset_index()
    u_posI_List.columns = ['user_id_idx', 'item_id_idx_list']
    return u_posI_List


def ignor_neg_item_list(train_pos_list_df, val_pos_list_df, test_pos_list_df, n_users):
    v = pd.merge(val_pos_list_df, test_pos_list_df, how='outer', left_on='user_id_idx', right_on='user_id_idx')
    t = pd.merge(train_pos_list_df, v, how='left', left_on='user_id_idx', right_on='user_id_idx')
    t.item_id_idx_list = t.item_id_idx_list.fillna('').apply(list)
    t.item_id_idx_list_x = t.item_id_idx_list_x.fillna('').apply(list)
    t.item_id_idx_list_x = t.item_id_idx_list_x.apply(lambda x: np.array(x) + n_users)
    t.item_id_idx_list_y = t.item_id_idx_list_y.fillna('').apply(list)
    t.item_id_idx_list_y = t.item_id_idx_list_y.apply(lambda x: np.array(x) + n_users)

    t['ignor_neg_list'] = [list((set(a).union(b).union(c))) for a, b, c in
                           zip(t.item_id_idx_list, t.item_id_idx_list_x, t.item_id_idx_list_y)]
    train_pos_list_df = t[['user_id_idx', 'item_id_idx_list', 'ignor_neg_list']]

    return train_pos_list_df


def prepare_val_test(train_df, val_df, test_df):
    r"""
    Sync nodes
    Relabelling user/ item nodes in 3 set.
    Unique identify user, item nodes in train set
    Add pos_item_list to 3 set.
    Add ignor_neg_list to train set

    Args:
        train_df/ val_df/ test_df (Tensor): Raw train_df/ val_df/ test_df

    Returns:
        n_users (int): number of unique users in train_df
        n_items (int): number of unique items in train_df
        train_df/ val_df/ test_df (Tensor): processed train_df/ val_df/ test_df

    """
    val_df, test_df = sync_nodes(train_df, val_df, test_df)
    n_users, n_items, train_df, val_df, test_df = relabelling(train_df, val_df, test_df)

    train_df['item_id_idx'] = train_df['item_id_idx'] + n_users

    train_pos_list_df = pos_item_list(train_df)
    val_pos_list_df = pos_item_list(val_df)
    test_pos_list_df = pos_item_list(test_df)

    train_pos_list_df = ignor_neg_item_list(train_pos_list_df, val_pos_list_df, test_pos_list_df, n_users)

    return n_users, n_items, train_df, train_pos_list_df, val_pos_list_df, test_pos_list_df


def df_to_graph(train_df, weight):
    r"""Convert dataset to bipartite graph_edge_index
    Args:
        :param train_df: (Tensor) Raw train_df
        :param weight: Graph contains weight or not
    Returns:

    """
    u_t = torch.LongTensor(train_df['user_id_idx'].values)
    i_t = torch.LongTensor(train_df['item_id_idx'].values)
    graph_edge_index = torch.stack((
        torch.cat([u_t, i_t]),
        torch.cat([i_t, u_t])
    ))

    if weight:
        w_t = torch.FloatTensor(train_df['weight'].values)
        edge_weights = torch.cat([w_t, w_t])
        return graph_edge_index, edge_weights

    return graph_edge_index


def sample_neg(x, n_neg, n_users, n_itm):
    """
    Need to add logic
    :param x:
    :param n_neg:
    :param n_users:
    :param n_itm:
    :return:
    """
    neg_list = list()
    while len(neg_list) < n_neg:
        neg_id = random.randint(0, n_itm-1) + n_users
        if neg_id not in x:
            neg_list.append(neg_id)
    return neg_list


def pos_neg_edge_index(train_pos_list_df, n_neg, n_users, n_itm):
    r"""Generate random neg_item for each (usr, pos_item) pair
    example: if n_neg=2
    train_df as below:
    user    pos_list
    u1      [1,2,3]
    u2      [7,8]

    Output should be(pos, neg item ids are + n_users):
    user    pos     neg
    u1      1       11
    u1      2       16
    u1      3       77
    u1      1       69
    u1      2       33
    u1      3       98
    u2      7       4
    u2      8       9
    u2      7       10
    u2      8       26

    Args:
        :param n_neg:
        :param train_pos_list_df: (Tensor)
        :param n_users: number of users
        :param n_itm: number of items

    Returns:
        users, pos_items, neg_items

    """
    u = [[a]*len(b)*n_neg for a, b in zip(train_pos_list_df.user_id_idx, train_pos_list_df.item_id_idx_list)]
    u = [item for sublist in u for item in sublist]   # flatten list of list
    users = torch.LongTensor(u)
    p = train_pos_list_df['item_id_idx_list'].apply(lambda x: x * n_neg).tolist()
    p = [item for sublist in p for item in sublist]
    pos_items = torch.LongTensor(p)
    n = [sample_neg(a, n_neg*len(b), n_users, n_itm) for a, b in
         zip(train_pos_list_df.ignor_neg_list, train_pos_list_df.item_id_idx_list)]
    n = [item for sublist in n for item in sublist]
    neg_items = torch.LongTensor(n)

    return users, pos_items, neg_items


def batch_pos_neg_edges(users, pos_items, neg_items):
    r"""Return (user+user, pos+neg) edge labels
    """
    users_label = torch.cat([users, users])
    pos_neg_items_label = torch.cat([pos_items, neg_items])
    batch_pos_neg_labels = torch.stack((users_label, pos_neg_items_label))
    return batch_pos_neg_labels


def interact_matrix(train_df, n_users, n_items):
    r"""
    create sparse tensor of all user-item interactions
    """
    i = torch.stack((
        torch.LongTensor(train_df['user_id_idx'].values),
        torch.LongTensor(train_df['item_id_idx'].values)
    ))
    v = torch.ones((len(train_df)), dtype=torch.float64)
    interactions_t = torch.sparse.FloatTensor(i, v, (n_users, n_items))
    return interactions_t


def get_metrics(user_Embed_wts, item_Embed_wts, test_pos_list_df, interactions_t, K):
    r"""
    Compute Precision@K, Recall@K
    # :param test_u_i_matrix:
    :param user_Embed_wts:
    :param item_Embed_wts:
    :param test_pos_list_df:
    :param K:
    :return: Recall@K, Precision@K
    """
    # prepare test set user list mask
    test_pos_list_df = test_pos_list_df.sort_values(by=['user_id_idx'])
    users = list(test_pos_list_df['user_id_idx'])

    # compute the score of aim_user-item pairs
    relevance_score = user_Embed_wts[users] @ item_Embed_wts.t()
    # mask out training user-item interactions from metric computation
    interactions_t = torch.index_select(interactions_t, 0, users)
    relevance_score = torch.mul(relevance_score, (1 - interactions_t))

    # compute top scoring items for each user
    r_cpu = relevance_score.cpu()
    topk_relevance_indices = torch.topk(r_cpu, K).indices
    topk_relevance_indices_df = pd.DataFrame(topk_relevance_indices.numpy())
    topk_relevance_indices_df['top_rlvnt_itm'] = topk_relevance_indices_df.values.tolist()
    topk_relevance_indices_df['user_ID'] = users
    topk_relevance_indices_df = topk_relevance_indices_df[['user_ID', 'top_rlvnt_itm']]

    # measure overlap between recommended (top-K) and held-out user-item interactions
    metrics_df = pd.merge(test_pos_list_df, topk_relevance_indices_df,
                          how='left', left_on='user_id_idx', right_on='user_ID')
    metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in
                                  zip(metrics_df.item_id_idx_list, metrics_df.top_rlvnt_itm)]  # TP

    metrics_df['recall'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / len(x['item_id_idx_list']), axis=1)
    metrics_df['precision'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / K, axis=1)

    return metrics_df['recall'].mean(), metrics_df['precision'].mean()


def regularization_loss(init_embed, batch_size, batch_usr, batch_pos, batch_neg, decay):
    r"""
    Compute loss from initial embeddings, used for regularization
    :param decay:
    :param init_embed: i.e. model.embedding.weight
    :param batch_size:
    :param batch_usr:
    :param batch_pos:
    :param batch_neg:
    :return: regularization loss
    """

    reg_loss = (1 / 2) * (
            init_embed[batch_usr].norm().pow(2) +
            init_embed[batch_pos].norm().pow(2) +
            init_embed[batch_neg].norm().pow(2)
    ) / batch_size

    return reg_loss * decay


def save_model(path, model, optimizer, precision, recall, epoch=None):
    """
    Function to save the trained model to disk.
    """

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H.%M.%S")

    torch.save({
                'timestamp': dt_string,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'precision': precision,
                'recall': recall
                }, path)

    print(f"Model {path} saved at {dt_string}")
