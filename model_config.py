import os
import json

# data
data_file = "nlu.json"

# BERT config file and ckpt file
bert_model_name = 'uncased_L-12_H-768_A-12'
bert_ckpt_dir = os.path.join("bert/", bert_model_name)
bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")

# Model path
model_version = 1
model_name = str(model_version)
model_save_dir = os.path.join(os.path.realpath("model"), model_name)

# hyper-parameters of the network
hp = open("params.txt", "r")
params = hp.read()
hp.close()
params = json.loads(params)
epochs = params['epoch']
batch_size = params['batch_size']
validation_split = params['val_split']
test_size = params['test_size']
embedding_dim_multiplier = params['embedding_multiplier']
learning_rate = params['lr']
dropout_ratio = params['dropout_ratio']
dense_units = params['units']
max_seq_len = params['max_seq_len']

# Visualization params
fig_size = (15, 8)
xticks_rotation = 45

