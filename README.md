# GNN-eCommerce
### Ecommerce Recommender using GNN
Team member: Ying Kang


## Introduction
Purpose of this project is to build a recommender system for eCommerce sites/apps. This system consists of:
1. A Graph Neural Network model. This model trains on past consumer events and makes product recommendations to consumers real time.
2. An automated pipeline to prepare train/validation/test datasets from raw data and train and evaluate the model.
3. An online model serving architecture: an api server and a model server to serve online store.


## Dataset
The dataset used to train and evaluate the model is eCommerce Events History in Cosmetics Shop:

https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop

We use DVC for version control and managing lineage. Remote repository is on Google Drive. The local "checked out" version is in data/raw/cosmetic-shop-ecommerce-events. 

data/raw/cosmetic-shop-ecommerce-events

