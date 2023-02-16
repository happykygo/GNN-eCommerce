import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.utils_v2 import *
from src.lightgcn import LightGCN
import yaml
import argparse


class TrainLightGCN:
    def __init__(self, csv_path, checkpoints_dir="model-checkpoints", samples=None):
        self.checkpoints_dir = checkpoints_dir
        self.csv_path = csv_path

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        interaction_matrix = pd.read_csv(self.csv_path)
        # todo remove later
        # interaction_matrix = interaction_matrix.rename(columns=({'product_id': 'item_id'}))

        if samples:
            interaction_matrix = interaction_matrix.sample(samples)

        train_df, test_df = train_test_split(interaction_matrix, test_size=0.05)
        test_df, val_df = train_test_split(test_df, test_size=0.5)

        self.n_users, self.n_items, train_df, self.train_pos_list_df, self.val_pos_list_df, \
        self.test_pos_list_df, self.val_interactions_t, self.test_interactions_t, val_df, test_df \
            = prepare_val_test(train_df, val_df, test_df)

        print("n_users : ", self.n_users, ", n_items : ", self.n_items)
        print("train_df Size  : ", len(train_df))
        print("val_pos_list_df Size : ", len(self.val_pos_list_df))
        print("test_pos_list_df Size : ", len(self.test_pos_list_df))

        self.edge_index, self.edge_weight = df_to_graph(train_df, True)
        self.edge_index = self.edge_index.to(self.device)
        self.edge_weight = self.edge_weight.to(self.device)

        train_df.to_csv(self.checkpoints_dir + "processed_train.csv")
        val_df.to_csv(self.checkpoints_dir + "processed_val.csv")
        test_df.to_csv(self.checkpoints_dir + "processed_test.csv")

    def __call__(self, *args, **kwargs):

        EPOCHS = int(args[0])

        tune_config = {
            "latent_dim": 80,
            "n_layers": 3,
            "LR": 0.005,
            "DECAY": 0.0001,  # reg loss
            "BATCH_SIZE": 1024,  # train mini batch size
        }

        model = LightGCN(self.n_users + self.n_items, tune_config["latent_dim"], tune_config["n_layers"])
        optimizer = torch.optim.Adam(model.parameters(), tune_config["LR"])

        K = 20  # Recall@K
        self.train(model, optimizer, EPOCHS=EPOCHS, BATCH_SIZE=tune_config["BATCH_SIZE"], K=K,
                   DECAY=tune_config["DECAY"], checkpoint_dir=self.checkpoints_dir)

        best_model = torch.load(self.checkpoints_dir + "/LightGCN_best.pt")
        best_epoch = best_model['epoch']
        best_val_precision = best_model['precision']
        best_val_recall = best_model['recall']

        test_model = LightGCN(self.n_users + self.n_items, tune_config["latent_dim"], tune_config["n_layers"])
        test_model.load_state_dict(best_model['model_state_dict'])

        # Evaluate model using test set
        test_p, test_recall = self.test(model, self.test_pos_list_df, self.test_interactions_t, K)

        print(
            f"Best epoch ({best_epoch}): Val Precision@{K}: {best_val_precision:>7f}, Recall@{K}: {best_val_recall:>7f}")
        print(f"Test Precision@{K}: {test_p:>7f}, Recall@{K}: {test_recall:>7f}")

    def train(self, model, optimizer, EPOCHS=50, BATCH_SIZE=1024, K=20, DECAY=0.0001, checkpoint_dir=""):

        model.to(self.device)

        bpr_loss_epoch_list = []
        reg_loss_epoch_list = []
        final_loss_epoch_list = []
        recall_epoch_list = []
        precision_epoch_list = []

        best_recall = 0.0

        print(f"Begin training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # for epoch in tqdm(range(EPOCHS)):
        for epoch in range(EPOCHS):
            users, pos_items, neg_items = pos_neg_edge_index(self.train_pos_list_df, self.n_users, self.n_items)
            # print(f"Total train set size = {len(users)}")
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)

            bpr_loss, reg_loss, final_loss = self.mini_batch_loop(users, pos_items, neg_items, model, optimizer,
                                                                  BATCH_SIZE, DECAY)

            precision, recall = self.test(model, self.val_pos_list_df, self.val_interactions_t, K)

            bpr_loss_epoch_list.append(bpr_loss)
            reg_loss_epoch_list.append(reg_loss)
            final_loss_epoch_list.append(final_loss)
            recall_epoch_list.append(recall)
            precision_epoch_list.append(precision)

            now = datetime.now().strftime('%H:%M:%S');
            print(f"Epoch {epoch} {now}: Val P@{K}: {precision:>7f}, R@{K}: {recall:>7f}, "
                  f"Train Loss: ({bpr_loss:>7f}, {reg_loss:>7f}, {final_loss:>7f})")

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

    def mini_batch_loop(self, users, pos_items, neg_items, model, optimizer, batch_size, decay):
        bpr_loss_batch_list = []
        reg_loss_batch_list = []
        final_loss_batch_list = []

        idx = list(range(len(users)))
        random.shuffle(idx)
        loader = DataLoader(idx, batch_size=batch_size, shuffle=True)

        model.train()
        # for batch_num, batch in enumerate(tqdm(loader)):
        for batch_num, batch in enumerate(loader):
            optimizer.zero_grad()

            batch_usr = users[batch]
            batch_pos_items = pos_items[batch]
            batch_neg_items = neg_items[batch]

            batch_pos_neg_labels = batch_pos_neg_edges(batch_usr, batch_pos_items, batch_neg_items)
            out = model(self.edge_index, batch_pos_neg_labels, self.edge_weight)
            size = len(batch)

            bpr_loss = model.recommendation_loss(out[:size], out[size:], 0) * size
            reg_loss = regularization_loss(model.embedding.weight, size, batch_usr, batch_pos_items, batch_neg_items,
                                           decay)
            loss = bpr_loss + reg_loss

            loss.backward()
            optimizer.step()

            bpr_loss_batch_list.append(bpr_loss.item())
            reg_loss_batch_list.append(reg_loss.item())
            final_loss_batch_list.append(loss.item())

        return np.mean(bpr_loss_batch_list), np.mean(reg_loss_batch_list), np.mean(final_loss_batch_list)

    def test(self, model, test_pos_list_df, interactions_t, k):
        model.eval()
        with torch.no_grad():
            embeds = model.get_embedding(self.edge_index, self.edge_weight)
            final_usr_embed, final_item_embed = torch.split(embeds, [self.n_users, self.n_items])
            test_topK_recall, test_topK_precision, _ = self.get_metrics(final_usr_embed, final_item_embed,
                                                                        test_pos_list_df, interactions_t, k)
        # user_id_list = list(test_pos_list_df['user_id_idx'])
        # with torch.no_grad():
        #     top_index_df = model.recommendK(self.edge_index, self.edge_weight, self.n_users, self.n_items,
        #                                     interactions_t, user_id_list, k)
        #     topK_recall, topK_precision, metrics = model.MARK_MAPK(test_pos_list_df, top_index_df, k)

        return test_topK_precision, test_topK_recall

    def get_metrics(self, user_Embed_wts, item_Embed_wts, test_pos_list_df, interactions_t, K):
        r"""
        Compute Precision@K, Recall@K
        # :param test_u_i_matrix:
        :param interactions_t:
        :param user_Embed_wts:
        :param item_Embed_wts:
        :param test_pos_list_df:
        :param K:
        :return: Recall@K, Precision@K
        """
        # prepare test set user list mask
        users = list(test_pos_list_df['user_id_idx'])
        # print(f"Test users: {len(users)}")

        # compute the score of aim_user-item pairs
        relevance_score = user_Embed_wts[users] @ item_Embed_wts.t()
        relevance_score = relevance_score.cpu()
        masked_relevance_score = torch.mul(relevance_score, (1 - interactions_t))
        # compute top scoring items for each user
        topk_relevance_indices = torch.topk(masked_relevance_score, K).indices
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
        metrics_df['precision'] = metrics_df.apply(
            lambda x: len(x['intrsctn_itm']) / K, axis=1)

        return metrics_df['recall'].mean(), metrics_df['precision'].mean(), metrics_df


def main(max_num_epochs=20, gpus_per_trial=1):
    with open("config.yaml") as config_file:
        config = yaml.safe_load(config_file)

    csv_path = config['data']['preprocessed'] + "u_i_weight_0.01_0.1_-0.09.csv"
    # file 0 -- interaction_matrix.csv
    # file 1 -- u_i_weight_0.01_0.1_-0.09.csv
    # file 2 -- u_i_weight_0.15_0.35_-0.2.csv
    checkpoint_dir = config['training']['checkpoints_dir']
    train_lightgcn = TrainLightGCN(csv_path, checkpoint_dir)
    train_lightgcn(max_num_epochs, gpus_per_trial)


if __name__ == "__main__":
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-e", "--epochs", required=True,
                    help="Max number of epochs")
    ap.add_argument("-g", "--gpus", required=True,
                    help="GPUs per trial")
    args = vars(ap.parse_args())

    main(args['epochs'], args['gpus'])
