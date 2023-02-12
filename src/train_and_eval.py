from utils_v2 import batch_pos_neg_edges, save_model, pos_neg_edge_index
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import yaml


class TrainLightGCN:
    def __int__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with open("params.yaml") as config_file:
            self.config = yaml.safe_load(config_file)

        interaction_matrix = pd.read_csv(self.config['data']['preprocessed'] + "interaction_matrix.csv")
        interaction_matrix = interaction_matrix.rename(columns={"product_id": "item_id"})
        self.interaction_matrix = interaction_matrix[['user_id', 'item_id', 'weight']]

        train_df, test_df = train_test_split(mini_im, test_size=0.1)
        test_df, val_df = train_test_split(test_df, test_size=0.5)



def train_and_evl(n_users, n_items, n_neg, edge_index, edge_weight, train_pos_list_df, test_pos_list_df, model, optimizer, device,
                  EPOCHS=50, BATCH_SIZE=1024, K=20, DECAY=0.0001, checkpoint_dir="", log_interval=10):  # test_u_i_matrix,

    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    model.to(device)

    bpr_loss_epoch_list = []
    reg_loss_epoch_list = []
    final_loss_epoch_list = []
    recall_epoch_list = []
    precision_epoch_list = []

    best_recall = 0.0

    for epoch in tqdm(range(EPOCHS)):
        users, pos_items, neg_items = pos_neg_edge_index(train_pos_list_df, n_neg, n_users, n_items)
        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        bpr_loss, reg_loss, final_loss = train_loop(users, pos_items, neg_items, edge_index, edge_weight, model,
                                                    optimizer, BATCH_SIZE)

        precision, recall = evaluation(model, n_users, n_items, edge_index, edge_weight, test_pos_list_df, K)

        bpr_loss_epoch_list.append(bpr_loss)
        reg_loss_epoch_list.append(reg_loss)
        final_loss_epoch_list.append(final_loss)
        recall_epoch_list.append(recall)
        precision_epoch_list.append(precision)

        print(f"Epoch {epoch}: Val P@{K}: {precision:>7f}, R@{K}: {recall:>7f}, Training Loss: ({bpr_loss:>7f}, {reg_loss:>7f}, {final_loss:>7f})")

        # save the best model
        if recall > best_recall:
            best_recall = recall
            save_model(checkpoint_dir + "/LightGCN_best.pt", model, optimizer, precision, recall, epoch=epoch)

    return (
        bpr_loss_epoch_list,
        reg_loss_epoch_list,
        final_loss_epoch_list,
        recall_epoch_list,
        precision_epoch_list)


def train_loop(users, pos_items, neg_items, edge_index, edge_weight, model, optimizer, BATCH_SIZE):
    bpr_loss_batch_list = []
    reg_loss_batch_list = []
    final_loss_batch_list = []

    idx = list(range(len(users)))
    random.shuffle(idx)
    loader = DataLoader(idx, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for batch_num, batch in enumerate(tqdm(loader)):
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

        loss.backward()
        optimizer.step()

        bpr_loss_batch_list.append(bpr_loss.item())
        reg_loss_batch_list.append(reg_loss.item())
        final_loss_batch_list.append(loss.item())

    return np.mean(bpr_loss_batch_list), np.mean(reg_loss_batch_list), np.mean(final_loss_batch_list)


def evaluation(model, n_users, n_items, edge_index, edge_weight, test_pos_list_df, K):  # test_u_i_matrix,
    model.eval()
    with torch.no_grad():
        embeds = model.get_embedding(edge_index, edge_weight)
        final_usr_embed, final_item_embed = torch.split(embeds, [n_users, n_items])
        test_topK_recall, test_topK_precision = get_metrics(final_usr_embed, final_item_embed, test_pos_list_df, K)  # test_u_i_matrix,

    return test_topK_precision, test_topK_recall


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