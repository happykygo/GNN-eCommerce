from sklearn import preprocessing as pp
import torch
import random
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader


def sync_nodes(train_df, val_df, test_df):
    r"""
    Remove user/items in val_df/test_df but not in train_df
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
    return val_df, test_df


def interact_matrix(train_df, n_users, n_items):
    r"""
    create dense tensor of all user-item interactions
    :param train_df: with 'user_id_idx' and 'item_id_idx'
    :param n_users:
    :param n_items:
    :return:
    """

    i = torch.stack((
        torch.LongTensor(train_df['user_id_idx'].values),
        torch.LongTensor(train_df['item_id_idx'].values)
    ))
    v = torch.ones((len(train_df)), dtype=torch.float64)
    interactions_t = torch.sparse.FloatTensor(i, v, (n_users, n_items)).to_dense()
    return interactions_t


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

    u_i_matrix = interact_matrix(train_df, n_users, n_items)
    train_df['item_id_idx'] = train_df['item_id_idx'] + n_users

    return n_users, n_items, train_df, val_df, test_df, u_i_matrix


def pos_item_list(train_df):
    r"""
    Generate Positive Item List
    :param train_df:
    :param val_df:
    :param test_df:
    :return:
    """
    u_i = pd.DataFrame(train_df, columns=['user_id_idx', 'item_id_idx', 'weight'])
    u_posI_List = train_df.loc[train_df['weight'] == 1]
    u_posI_List = u_posI_List.groupby('user_id_idx')['item_id_idx'].apply(list).reset_index()
    u_posI_List.columns = ['user_id_idx', 'item_id_idx_list']
    u_posI_List = pd.merge(u_posI_List, u_i, how='right', left_on='user_id_idx', right_on='user_id_idx')
    return u_posI_List


def prepare_val_test(train_df, val_df, test_df):
    r"""Relabelling user/ item nodes.
        Remove user/items in val_df/test_df but not in train_df

    Args:
        train_df/ val_df/ test_df (Tensor): Raw train_df/ val_df/ test_df

    Returns:
        n_users (int): number of unique users in train_df
        n_items (int): number of unique items in train_df
        train_df/ val_df/ test_df (Tensor): processed train_df/ val_df/ test_df

    """
    val_df, test_df = sync_nodes(train_df, val_df, test_df)
    n_users, n_items, train_df, val_df, test_df, u_i_matrix = relabelling(train_df, val_df, test_df)
    train_df = pos_item_list(train_df)
    val_df = pos_item_list(val_df)
    test_df = pos_item_list(test_df)

    return n_users, n_items, train_df, val_df, test_df, u_i_matrix


def df_to_graph(train_df, weight):
    r"""Convert dataset to bipartite graph_edge_index
        return 2 * 132032, for model.forward
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

    return graph_edge_index, edge_weights if weight else graph_edge_index


def pos_neg_edge_index(train_df, n_users, n_itm):
    r"""Generate random neg_item for each (usr, pos_item) pair
    Args:
        :param train_df: (Tensor)
        :param n_users: number of users
        :param n_itm: number of items

    Returns:
        users, pos_items, neg_items

    """
    def sample_neg(x):
        while True:
            neg_id = random.randint(0, n_itm - 1)
            if neg_id not in x:
                return neg_id

    train_df = train_df.loc[train_df['weight'] == 1]
    users = torch.LongTensor(train_df['user_id_idx'].values)
    pos_items = torch.LongTensor(train_df['item_id_idx'].values)
    neg_items = torch.LongTensor(train_df['item_id_idx_list'].apply(lambda x: sample_neg(x)).values+n_users)

    return users, pos_items, neg_items


def batch_pos_neg_edges(users, pos_items, neg_items):
    r"""Return (user+user, pos+neg) edge labels

    Args:
        users, pos_items, neg_items

    """
    users_label = torch.cat([users, users])
    pos_neg_items_label = torch.cat([pos_items, neg_items])
    batch_pos_neg_labels = torch.stack((users_label, pos_neg_items_label))
    return batch_pos_neg_labels


def regularization_loss(init_embed, batch_size, batch_usr, batch_pos, batch_neg,
                        DECAY=0.0001):
    r"""
    Compute loss from initial embeddings, used for regularization
    :param init_embed: i.e. model.embedding.weight
    :param batch_size:
    :param batch_usr:
    :param batch_pos:
    :param batch_neg:
    :param DECAY: DEFAULT 0.0001
    :return: regularization loss
    """

    reg_loss = (1 / 2) * (
            init_embed[batch_usr].norm().pow(2) +
            init_embed[batch_pos].norm().pow(2) +
            init_embed[batch_neg].norm().pow(2)
    ) / batch_size

    return reg_loss * DECAY


def train_loop(n_users, n_items, train_df, edge_index, edge_weight, model, optimizer, BATCH_SIZE):
    bpr_loss_batch_list = []
    reg_loss_batch_list = []
    final_loss_batch_list = []

    users, pos_items, neg_items = pos_neg_edge_index(train_df, n_users, n_items)

    idx = list(range(len(users)))
    random.shuffle(idx)
    loader = DataLoader(idx, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for batch in loader:
        optimizer.zero_grad()

        batch_usr = users[batch]
        batch_pos_items = pos_items[batch]
        batch_neg_items = neg_items[batch]

        batch_pos_neg_labels = batch_pos_neg_edges(batch_usr, batch_pos_items, batch_neg_items)
        out = model(edge_index, batch_pos_neg_labels, edge_weight)
        size = len(batch)

        bpr_loss = model.recommendation_loss(out[:size], out[size:], 0) * size
        reg_loss = regularization_loss(model.embedding.weight, size, batch_usr, batch_pos_items, batch_neg_items)
        loss = bpr_loss + reg_loss

        # print("bpr loss: ", loss, "reg loss: ", reg_loss)
        loss.backward()
        optimizer.step()

        bpr_loss_batch_list.append(bpr_loss.item())
        reg_loss_batch_list.append(reg_loss.item())
        final_loss_batch_list.append(loss.item())

    bpr_loss = round(np.mean(bpr_loss_batch_list), 4)
    reg_loss = round(np.mean(reg_loss_batch_list), 4)
    final_loss = round(np.mean(final_loss_batch_list), 4)
    return bpr_loss, reg_loss, final_loss


def get_metrics(user_Embed_wts, item_Embed_wts, u_i_matrix, test_data, K):
    r"""
    Need review!!! Slow!!!
    Compute Precision@K, Recall@K
    :param u_i_matrix:
    :param user_Embed_wts:
    :param item_Embed_wts:
    :param test_data:
    :param K:
    :return: Recall@K, Precision@K
    """

    # compute the score of all user-item pairs
    relevance_score = user_Embed_wts @ item_Embed_wts.t()

    # mask out training user-item interactions from metric computation
    relevance_score = torch.mul(relevance_score, (1 - u_i_matrix))

    # compute top scoring items for each user
    topk_relevance_indices = torch.topk(relevance_score, K).indices
    topk_relevance_indices_df = pd.DataFrame(topk_relevance_indices.cpu().numpy())
    topk_relevance_indices_df['top_rlvnt_itm'] = topk_relevance_indices_df.values.tolist()
    topk_relevance_indices_df['user_ID'] = topk_relevance_indices_df.index
    topk_relevance_indices_df = topk_relevance_indices_df[['user_ID', 'top_rlvnt_itm']]

    # measure overlap between recommended (top-K) and held-out user-item interactions
    test_interacted_items = test_data.groupby('user_id_idx')['item_id_idx'].apply(list).reset_index()
    metrics_df = pd.merge(test_interacted_items, topk_relevance_indices_df,
                          how='left', left_on='user_id_idx', right_on=['user_ID'])
    metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in
                                  zip(metrics_df.item_id_idx, metrics_df.top_rlvnt_itm)]  # TP

    metrics_df['recall'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / len(x['item_id_idx']), axis=1)
    metrics_df['precision'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / K, axis=1)

    return metrics_df['recall'].mean(), metrics_df['precision'].mean()


def evaluation(model, n_users, n_items, edge_index, edge_weight, u_i_matrix, val_df, K):
    model.eval()
    with torch.no_grad():
        embeds = model.get_embedding(edge_index, edge_weight)
        final_usr_embed, final_item_embed = torch.split(embeds, (n_users, n_items))
        test_topK_recall, test_topK_precision = \
            get_metrics(final_usr_embed, final_item_embed, u_i_matrix, val_df, K)

    precision = round(test_topK_precision, 4)
    recall = round(test_topK_recall, 4)
    return precision, recall
