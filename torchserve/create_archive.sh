MODEL_CHECKPOINT_PATH="../model-checkpoints/2023-02-18_071308"

torch-model-archiver --model-name lightgcn_recommender \
--version 0.1 \
--model-file lightgcn.py \
--serialized-file $MODEL_CHECKPOINT_PATH/LightGCN_best.pt \
--handler lightgcn_handler.py \
-r requirements.txt \
--extra-files $MODEL_CHECKPOINT_PATH/processed_train.csv
