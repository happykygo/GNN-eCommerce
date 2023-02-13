from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from utils_v2 import *
import yaml


class TrainLightGCN:
    def __int__(self, csv_file_name):  # "interaction_matrix.csv"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with open("params.yaml") as config_file:
            config = yaml.safe_load(config_file)

        interaction_matrix = pd.read_csv(config['data']['preprocessed']+csv_file_name)

        train_df, test_df = train_test_split(interaction_matrix, test_size=0.1)
        test_df, val_df = train_test_split(test_df, test_size=0.5)

        self.n_users, self.n_items, self.train_df, self.train_pos_list_df, self.val_pos_list_df, self.test_pos_list_df \
            = prepare_val_test(train_df, val_df, test_df)
        self.interactions_t = interact_matrix(self.train_df, self.n_users, self.n_items)
        self.edge_index, self.edge_weight = df_to_graph(train_df, True)
        self.edge_index = self.edge_index.to(self.device)
        self.edge_weight = self.edge_weight.to(self.device)

    def __call__(self, *args, **kwargs):
        self.__int__('')

    def train_and_evl(self, model, optimizer, n_neg=1, EPOCHS=50, BATCH_SIZE=1024, K=20,
                      DECAY=0.0001, checkpoint_dir="", log_interval=10):

        model.to(self.device)

        bpr_loss_epoch_list = []
        reg_loss_epoch_list = []
        final_loss_epoch_list = []
        recall_epoch_list = []
        precision_epoch_list = []

        best_recall = 0.0

        for epoch in tqdm(range(EPOCHS)):
            users, pos_items, neg_items = pos_neg_edge_index(self.train_pos_list_df, n_neg, self.n_users, self.n_items)
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)

            bpr_loss, reg_loss, final_loss = self.train_loop(users, pos_items, neg_items, model, optimizer, BATCH_SIZE, DECAY)

            precision, recall = self.test(model, self.val_pos_list_df, K)

            bpr_loss_epoch_list.append(bpr_loss)
            reg_loss_epoch_list.append(reg_loss)
            final_loss_epoch_list.append(final_loss)
            recall_epoch_list.append(recall)
            precision_epoch_list.append(precision)

            print(f"Epoch {epoch}: Val P@{K}: {precision:>7f}, R@{K}: {recall:>7f}, "
                  f"Training Loss: ({bpr_loss:>7f}, {reg_loss:>7f}, {final_loss:>7f})")

            # save the best model
            if recall > best_recall:
                best_recall = recall
                save_model(checkpoint_dir+"/LightGCN_best.pt", model, optimizer, precision, recall, epoch=epoch)

        return (
            bpr_loss_epoch_list,
            reg_loss_epoch_list,
            final_loss_epoch_list,
            recall_epoch_list,
            precision_epoch_list)

    def train_loop(self, users, pos_items, neg_items, model, optimizer, batch_size, decay):
        bpr_loss_batch_list = []
        reg_loss_batch_list = []
        final_loss_batch_list = []

        idx = list(range(len(users)))
        random.shuffle(idx)
        loader = DataLoader(idx, batch_size=batch_size, shuffle=True)

        model.train()
        for batch_num, batch in enumerate(tqdm(loader)):
            optimizer.zero_grad()

            batch_usr = users[batch]
            batch_pos_items = pos_items[batch]
            batch_neg_items = neg_items[batch]

            batch_pos_neg_labels = batch_pos_neg_edges(batch_usr, batch_pos_items, batch_neg_items)
            out = model(self.edge_index, batch_pos_neg_labels, self.edge_weight)
            size = len(batch)

            bpr_loss = model.recommendation_loss(out[:size], out[size:], 0) * size
            reg_loss = regularization_loss(model.embedding.weight, size, batch_usr, batch_pos_items, batch_neg_items, decay)
            loss = bpr_loss + reg_loss

            loss.backward()
            optimizer.step()

            bpr_loss_batch_list.append(bpr_loss.item())
            reg_loss_batch_list.append(reg_loss.item())
            final_loss_batch_list.append(loss.item())

        return np.mean(bpr_loss_batch_list), np.mean(reg_loss_batch_list), np.mean(final_loss_batch_list)

    def test(self, model, test_pos_list_df, K):
        model.eval()
        with torch.no_grad():
            embeds = model.get_embedding(self.edge_index, self.edge_weight)
            final_usr_embed, final_item_embed = torch.split(embeds, [self.n_users, self.n_items])
            test_topK_recall, test_topK_precision = get_metrics(final_usr_embed, final_item_embed,
                                                                test_pos_list_df, self.interactions_t, K)

        return test_topK_precision, test_topK_recall
