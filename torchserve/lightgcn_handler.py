import torch
from ts.torch_handler.base_handler import BaseHandler
import pandas as pd
import os
from lightgcn import LightGCN


class LightGCNHandler(BaseHandler):
    """
    A custom model handler implementation.
    """
    def initialize(self, context):
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """

        #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:"+str(properties.get("gpu_id"))) if torch.cuda.is_available() else "cpu"

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        train_df_path = os.path.join(model_dir, 'processed_train.csv')
        train_df = pd.read_csv(train_df_path)
        self.n_users = train_df['user_id_idx'].nunique()
        self.n_items = train_df['item_id_idx'].nunique()
        train_df['item_id_idx'] = train_df['item_id_idx'] - self.n_users
        self.i_m_matrix = self.interact_matrix(train_df, self.n_users, self.n_items)  # keep in cpu
        self.edge_index, self.edge_weight = self.df_to_graph(train_df, True)

        self.edge_index = self.edge_index.to(self.device)
        self.edge_weight = self.edge_weight.to(self.device)
        self.i_m_matrix = self.i_m_matrix.to(self.device)

        model_state = torch.load(model_pt_path)
        hyperparams = model_state['hyperparams']
        self.model = LightGCN(self.n_users + self.n_items, hyperparams['latent_dim'], hyperparams['n_layers'])
        self.model.load_state_dict(model_state['model_state_dict'])
        self.model.to(self.device)

        self.initialized = True

    def preprocess(self, data):
        """
        Preprocess function to convert the request input to a tensor(Torchserve supported format).
        The user needs to override to customize the pre-processing

        Args :
            data (list): List of the data from the request input.

        Returns:
            tensor: Returns the tensor data of the input
        """
        print("-----------------------------")
        print("data: ", data)
        preprocessed_data = data[0].get("data")
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")
        print("preprocessed_data: ", preprocessed_data)
        print("-----------------------------")

        return preprocessed_data
    
    def inference(self, data, *args, **kwargs):
        """The Inference Request is made through this function and the user
        needs to override the inference function to customize it.

        Args:
            data (torch tensor): The data is in the form of Torch Tensor
                                 whose shape should match that of the
                                  Model Input shape.

        Returns:
            (Torch Tensor): The predicted response from the model is returned
                            in this function.
        """
        with torch.no_grad():
            marshalled_data = torch.as_tensor(data, device=self.device)
            interactions_t = torch.index_select(self.i_m_matrix, 0, marshalled_data).to_dense().cpu()

            k = 20
            topK_df = self.model.recommendK(self.edge_index, self.edge_weight, self.n_users, self.n_items,
                                            interactions_t, data, k)

            topK = topK_df['top_rlvnt_itm']

        return topK

    def df_to_graph(self, train_df, weight):
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

    def interact_matrix(self, train_df, n_users, n_items):
        r"""
        create sparse tensor of all user-item interactions
        """
        df = train_df.loc[train_df['weight'] == 1.0]
        i = torch.stack((
            torch.LongTensor(df['user_id_idx'].values),
            torch.LongTensor(df['item_id_idx'].values)
        ))
        v = torch.ones((len(df)), dtype=torch.float32)
        interactions_t = torch.sparse.FloatTensor(i, v, (n_users, n_items))
        return interactions_t
