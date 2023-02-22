from sklearn.model_selection import train_test_split
from utils_v2 import *
from lightgcn import LightGCN
import yaml
import argparse


class TrainLightGCN:
    def __init__(self, csv_path, checkpoints_dir="model-checkpoints", gpu=0, samples=None):
        self.checkpoints_dir = checkpoints_dir
        self.csv_path = csv_path

        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        print(f"device: {self.device}")
        
        interaction_matrix = pd.read_csv(self.csv_path)

        if samples:
            interaction_matrix = interaction_matrix.sample(samples)

        train_df, test_df = train_test_split(interaction_matrix, test_size=0.05)
        test_df, val_df = train_test_split(test_df, test_size=0.5)

        self.n_users, self.n_items, train_df, self.train_pos_list_df, self.val_pos_list_df, \
        self.test_pos_list_df, self.val_interactions_t, self.test_interactions_t, val_df, test_df \
            = prepare_val_test(train_df, val_df, test_df)

        print("n_users : ", self.n_users, ", n_items : ", self.n_items)
        print("train_df Size  : ", len(train_df))
        print("train_pos_list_df Size : ", len(self.train_pos_list_df))
        print("val_pos_list_df Size : ", len(self.val_pos_list_df))
        print("test_pos_list_df Size : ", len(self.test_pos_list_df))

        self.train_size = len(train_df)
        self.edge_index, self.edge_weight = df_to_graph(train_df, True)
        self.edge_index = self.edge_index.to(self.device)
        self.edge_weight = self.edge_weight.to(self.device)

        train_df.to_csv(self.checkpoints_dir + "processed_train.csv")
        val_df.to_csv(self.checkpoints_dir + "processed_val.csv")
        test_df.to_csv(self.checkpoints_dir + "processed_test.csv")

    def __call__(self, *args, **kwargs):

        EPOCHS = int(args[0])

        self.tune_config = tune_config = {
            "latent_dim": 96,
            "n_layers": 5,
            "LR": 0.005,
            "DECAY": 0.0001,  # reg loss
            "BATCH_SIZE": 1024,  # train mini batch size
        }

        print(tune_config)

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
        test_p, test_recall, _ = self.test(model, self.test_pos_list_df, self.test_interactions_t, K)

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
        n_batch = int(self.train_size / (BATCH_SIZE*40))
        print(f"1 epoch = {n_batch} batches")
        
        # for epoch in tqdm(range(EPOCHS)):
        for epoch in range(EPOCHS):
            bpr_loss, reg_loss, final_loss = self.mini_batch_loop(model, optimizer, BATCH_SIZE, DECAY, n_batch)

            precision, recall, _ = self.test(model, self.val_pos_list_df, self.val_interactions_t, K)

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
                save_model(checkpoint_dir + "/LightGCN_best.pt", model, optimizer, precision, recall, epoch=epoch, hyperparams=self.tune_config)

        return (
            bpr_loss_epoch_list,
            reg_loss_epoch_list,
            final_loss_epoch_list,
            recall_epoch_list,
            precision_epoch_list)

    def mini_batch_loop(self, model, optimizer, batch_size, decay, n_batch):
        bpr_loss_batch_list = []
        reg_loss_batch_list = []
        final_loss_batch_list = []

        model.train()
        for batch_idx in range(n_batch):
            optimizer.zero_grad()

            users, pos_items, neg_items = batch_loader(self.train_pos_list_df, batch_size, self.n_users, self.n_items)
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)

            batch_pos_neg_labels = batch_pos_neg_edges(users, pos_items, neg_items)
            out = model(self.edge_index, batch_pos_neg_labels, self.edge_weight)
            size = len(users)

            bpr_loss = model.recommendation_loss(out[:size], out[size:], 0) * size
            reg_loss = regularization_loss(model.embedding.weight, size, users, pos_items, neg_items,
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
        user_id_list = list(test_pos_list_df['user_id_idx'])
        with torch.no_grad():
            top_index_df = model.recommendK(self.edge_index, self.edge_weight, self.n_users, self.n_items,
                                            interactions_t, user_id_list, k)
            topK_precision, topK_recall, metrics = model.MARK_MAPK(test_pos_list_df, top_index_df, k)
        return topK_precision, topK_recall, metrics


def main(max_num_epochs=20, gpu=0):
    with open("config.yaml") as config_file:
        config = yaml.safe_load(config_file)

    csv_path = config['data']['preprocessed'] + "u_i_weight_0.01_0.1_-0.09.csv"
    # file 0 -- interaction_matrix.csv
    # file 1 -- u_i_weight_0.01_0.1_-0.09.csv
    # file 2 -- u_i_weight_0.15_0.35_-0.2.csv
    checkpoint_dir = config['training']['checkpoints_dir']
    train_lightgcn = TrainLightGCN(csv_path, checkpoint_dir, gpu)
    train_lightgcn(max_num_epochs)


if __name__ == "__main__":
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-e", "--epochs", required=True,
                    help="Max number of epochs")
    ap.add_argument("-g", "--gpus", required=True,
                    help="GPUs per trial")
    args = vars(ap.parse_args())

    main(args['epochs'], args['gpu'])
