from src.lightgcn import LightGCN
from src.utils_v2 import interact_matrix, df_to_graph, load_data_model, pos_item_list
import pandas as pd
import torch
import yaml


class InferenceLightGCN:
    def __init__(self, checkpoint_dir, gpu):
        # train_df = pd.read_csv(checkpoint_dir + 'processed_train.csv')  # nodes are uniquely identified
        # test_df = pd.read_csv(checkpoint_dir + 'processed_test.csv')
        # val_df = pd.read_csv(checkpoint_dir + 'processed_val.csv')
        # if gpu:
        #     best_model = torch.load(checkpoint_dir + "LightGCN_best.pt", map_location=torch.device('cpu'))
        # else:
        #     best_model = torch.load(checkpoint_dir + "LightGCN_best.pt")

        train_df, test_df, val_df, best_model = load_data_model(checkpoint_dir, gpu)
        self.edge_index, self.edge_weight = df_to_graph(train_df, True)

        n_users = train_df['user_id_idx'].nunique()
        n_items = train_df['item_id_idx'].nunique()
        train_df['item_id_idx'] = train_df['item_id_idx'] - n_users
        i_m_matrix = interact_matrix(train_df, n_users, n_items)  # keep in cpu

        val_test = pd.concat([test_df, val_df], ignore_index=True)
        val_test = pos_item_list(val_test).sort_values(by=['user_id_idx'])
        user_list = list(val_test['user_id_idx'])
        self.interactions_t = torch.index_select(i_m_matrix, 0, torch.tensor(user_list)).to_dense()

        # users_list = self.combined[['user_id', 'user_id_idx']].drop_duplicates()
        # purchased_users = val_test.loc[val_test['weight'] == 1.0]
        # self.users_list = users_list[['user_id', 'user_id_idx']].drop_duplicates()

        self.test_model = LightGCN(n_users + n_items, 80, 3)
        self.test_model.load_state_dict(best_model['model_state_dict'])

        self.user_list = user_list
        self.n_users = n_users
        self.n_items = n_items
        self.val_test = val_test

    def recommendation(self, k):
        topK_df = self.test_model.recommendK(self.edge_index, self.edge_weight, self.n_users, self.n_items,
                                          self.interactions_t, self.user_list, k)
        result = pd.merge(self.val_test, topK_df, how='left', left_on='user_id_idx', right_on='user_ID')
        result['overlap_item'] = [list(set(a).intersection(b)) for a, b in zip(result.item_id_idx_list, result.top_rlvnt_itm)]
        print(f"Result: \n {result.to_string()}")

        return result



def main():
    with open("config.yaml") as config_file:
        config = yaml.safe_load(config_file)

    checkpoint_dir = config['training']['checkpoints_dir']+'2023-02-15_060043/'
    inferenceModel = InferenceLightGCN(checkpoint_dir, True)

    # target_users = list(inferenceModel.users_list['user_id_idx'].sample(1))
    # Rec for purchased user
    # target_users = list(inferenceModel.p_user_list['user_id_idx'].sample(1))
    k = 20
    result = inferenceModel.recommendation(k)
    inference_dir = config['inference']['model-recommendations/']
    result.to_csv(inference_dir+'K'+str(k)+'-2023-02-15_060043.csv')

    # target_users = list([67, 96])
    # t_result = result.loc[(result['user_id_idx'].isin(target_users))]
    # print(f'Target users are : {target_users}; \nRecommendation for user: \n {t_result.to_string()}')

if __name__ == "__main__":
    main()

