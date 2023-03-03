# GNN-eCommerce
### Ecommerce Recommender using GNN
Team member: Ying Kang

    - Document limitations of your model / data / ML pipeline

## Introduction -- need check

Purpose of this project is to build a recommender system for eCommerce sites/apps. 
The recommender is based on recent research paper, by [He et al. 2020](https://arxiv.org/abs/2002.02126). It applied Graph Neural Network on the multi-form user-product interactions business problem. 

This system consists of:
1. A Graph Neural Network model. This model trains on historical consumer events and makes product recommendations to consumers [?real time?]().
2. An automated pipeline to prepare train/validation/test datasets from raw data and train and evaluate the model.
3. An online model serving architecture: an api server and a model server to serve online store.

## Use cases

This system is suitable for recommending products for consumers who have buying or viewing history. 

For new customers who have no history activity records, it should be handled by other models, i.e. Popularity recommender, Random recommender.

For consumers' in-session activity, these can be caught by a content filtering recommender and combine the result with this GNN-recommender.

?? Training the model with dataset mentioned above costs about 24 hours. The model can be iteratively re-trained and re-deployed periodically. ??


## Dataset and [Data pipline](notebooks/1.data_preprocessing.ipynb)
The dataset used to train and evaluate the model is eCommerce Events History in Cosmetics Shop from [Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop).
It contains 20 million events data for 1.6 million consumers interacted with 54 thousand items during 5 months.

DVC is used for version control and managing lineage. Remote repository is on Google Drive. The local "checked out" version is in [directory](data/raw/cosmetic-shop-ecommerce-events). 

There are multiple steps and ways to transform raw data into graph representation.

Here is a summary:
1. Map each `eventType` to related `eventWeight`. This is a config setting. I tried out 2 different reasonable possible configs, and it turns out this does not affect the model performance much.
2. Sum up multiple `eventWeight`(`eventType`) between a consumer-product pair to be raw `edgeWeight`.
3. Adjust raw `edgeWeight` to a proper range (between 0 and 1).
   1. Negative raw `edgeWeight` is adjusted to be equal to `eventWeight` of _View_.
   2. raw `edgeWeight` greater than 1 and contains at least **one** _Purchase_ `eventType`, this is adjusted to be 1.0.
   3. raw `edgeWeight` greater than 1 and contains **NO** _Purchase_ `eventType`, this is adjusted to be 0.5.

Process data is persisted in [directory](data/preprocessed/u_i_weight_0.01_0.1_-0.09.csv).

## [Model Architecture](src/lightgcn.py)
The model architecture is implemented using PyG LightGCN based on research paper mentioned above. Some features are adjusted and added to form the customized model architecture to fulfill the needs of this project.
1. Support `edgeWeight` feature in the model architecture.
2. Recommend K items for given users
3. Compute `MARK`(Mean Avg. Recall @ K) and `MAPK`(Mean Avg. Precision @ K)
4. Fix `BPR_loss` bug.

## [Train and save the best model](src/train_lightgcn.py)

### [Prepare train/val/test dataset](src/utils_v2.py) for training and testing

The model performance is evaluated by `MARK`. To hold the ground_truth for evaluation purpose, I split the whole dataset into train/ val/ test. Sync up `users`, `items` nodes in three dataset(`users`, `items` nodes that only exist in val/ test set are not usable for evaluation).

### Mini-batch training

The model is trained in mini-batch fashion to optimize the BPR-loss. Each mini-batch contains `batch_size` (user : positive_item : negative_item) pairs which serves `BPR-loss` calculation.

**Tunable hyper parameters** are: `n_layers`, `latent_dimension`, `learning_rate`

### Save best model
The training process keeps track of best model and related metadata. Saves the best model and its metadata in [directory](model-checkpoints/).

## [Inference the model](src/inference_lightgcn.py)
The inference process loads saved best model and its metadata, recommend topK products, compute `MARK`, compute paths between users and recommended products using networkX `shortest_path`, and persist the result in [directory](model-recommendations).

Persisted inference result can be plot out: [Result explainability](notebooks/plot_inference_result.ipynb), [Plot utils](src/plot.py)

## MLE software architecture

[MLOps Stack](doc/MLOps%20Stack.jpeg)
[Product Infrastructure](doc/GNN-eCommerce%20Infrastructure.jpeg)
[TorchServe](torchserve/lightgcn_handler.py)

??? When using this model to serve an online eCommerce application(web or app), ??? mentioned lightgcn_handle.py a little ???


## Future Work
The current model is optimized on the probability of buying. There may be other optimization metric such as revenue, etc. We can try ensemble of models with different optimization metrics.

Another possibility is to add price bucket in the graph.

Groping events in session is also worth exploring.


## [Project Pitch Slides](doc/GNN-based%20RecSys.pdf)
