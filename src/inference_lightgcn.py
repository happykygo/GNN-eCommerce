import argparse
import torch
import pandas as pd
import yaml
from src.lightgcn import LightGCN
from src.utils_v2 import interact_matrix, df_to_graph


class InferenceLightGCN:
    def __init__(self, checkpoint_dir):
        train_df = pd.read_csv(checkpoint_dir + 'processed_train.csv')
        test_df = pd.read_csv(checkpoint_dir + 'processed_test.csv')
        val_df = pd.read_csv(checkpoint_dir + 'processed_val.csv')
        best_model = torch.load(checkpoint_dir + "LightGCN_best.pt")

        self.edge_index, self.edge_weight = df_to_graph(train_df, True)

        self.n_users = train_df['user_id_idx'].nunique()
        self.n_items = train_df['item_id_idx'].nunique()
        # note: item_id in train_df need to -n_users before use it!!
        train_df['item_id_idx'] = train_df['item_id_idx'] - self.n_users
        self.combined = pd.concat([train_df, test_df, val_df], ignore_index=True)
        self.interactions_t = interact_matrix(self.combined, self.n_users, self.n_items)

        self.test_model = LightGCN(self.n_users + self.n_items, 80, 3)
        self.test_model.load_state_dict(best_model['model_state_dict'])

        self.users_list = self.combined[['user_id', 'user_id_idx']]
        self.users_list = self.users_list.drop_duplicates()

    def recommendation(self, user_id_list, k):
        test_interactions_t = torch.index_select(self.interactions_t, 0, torch.tensor(user_id_list)).to_dense()

        df = self.combined
        df = df.loc[(df['user_id_idx'].isin(user_id_list))]  #  & (df['weight'] == 1.0)
        print(f"Real data: \n {df}")
        return self.test_model.recommendK(self.edge_index, self.edge_weight, self.n_users, self.n_items,
                                          test_interactions_t, user_id_list, k)


def main():
    with open("config.yaml") as config_file:
        config = yaml.safe_load(config_file)

    checkpoint_dir = config['training']['checkpoints_dir']
    inferenceModel = InferenceLightGCN(checkpoint_dir)

    target_users = list(inferenceModel.users_list['user_id_idx'].sample(1))

    top_index = inferenceModel.recommendation(target_users, k=5)

    print(f'Target users are : {target_users}; \nRecommendation for user: {top_index}')


if __name__ == "__main__":
    # Construct the argument parser
    # ap = argparse.ArgumentParser()
    #
    # # Add the arguments to the parser
    # ap.add_argument("-g", "--gpus", required=True,
    #                 help="GPUs per trial")
    # args = vars(ap.parse_args())

    main()
