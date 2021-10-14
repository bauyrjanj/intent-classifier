import pandas as pd
from sklearn.model_selection import train_test_split
from bert.tokenization.bert_tokenization import FullTokenizer
import os
import tensorflow as tf
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from tensorflow import keras
import keras_tuner as kt
import json
from tqdm import tqdm
import numpy as np

# BERT config file and ckpt file
bert_model_name = 'uncased_L-12_H-768_A-12'
bert_ckpt_dir = os.path.join("bert/", bert_model_name)
bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")

# Hardcoded hyper-parameters
embedding_dim_multiplier = 2
val_split = 0.3
test_size = 0.3
batch_size=16
random_state = 42

df = pd.read_json('nlu.json', orient='table')
classes = df['intent'].unique().tolist()
X = df['text']
y = df['intent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
frame = {'text': X_train, 'intent': y_train}
train = pd.DataFrame(frame)
frame = {'text': X_test, 'intent': y_test}
test = pd.DataFrame(frame)

class preproc_data:
    DATA_COLUMN = "text"
    LABEL_COLUMN = "intent"
    def __init__(self, train, test, tokenizer: FullTokenizer, classes, max_seq_len=300):
        self.tokenizer = tokenizer
        self.max_seq_len = 0
        self.classes = classes

        ((self.train_x, self.train_y), (self.test_x, self.test_y)) = map(self._prepare, [train, test])

        self.max_seq_len = min(self.max_seq_len, max_seq_len)
        self.train_x, self.test_x = map(self._pad, [self.train_x, self.test_x])
    def _prepare(self, df: pd):
        x, y = [], []
        for _, row in tqdm(df.iterrows()):
            text, label = row[preproc_data.DATA_COLUMN], row[preproc_data.LABEL_COLUMN]
            tokens = self.tokenizer.tokenize(text)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            self.max_seq_len = max(self.max_seq_len, len(token_ids)*embedding_dim_multiplier)
            x.append(token_ids)
            y.append(self.classes.index(label))
        return np.array(x), np.array(y)
    def _pad(self, ids):
        x = []
        for input_ids in ids:
            input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
            x.append(np.array(input_ids))
        return np.array(x)

tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
data = preproc_data(train, test, tokenizer, classes)
max_seq_len = data.max_seq_len


def build_model(hp):
    with tf.io.gfile.GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        bert = BertModelLayer.from_params(bert_params, name="bert")
    input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
    bert_output = bert(input_ids)
    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)

    hp_dout = hp.Choice('dropout_ratio', values=[0.3, 0.5])
    cls_out = keras.layers.Dropout(hp_dout)(cls_out)

    hp_units = hp.Int('units', min_value=768, max_value=1532, step=96)
    logits = keras.layers.Dense(units=hp_units, activation="relu")(cls_out)

    logits = keras.layers.Dropout(hp_dout)(logits)
    logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)
    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))
    load_stock_weights(bert, bert_ckpt_file)

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-5])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=50,
                     factor=3,
                     directory='my_dir',
                     project_name='hp_tuning')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

num_batches = int(data.train_x.shape[0]/batch_size)
batch_start = 0
batch_end = batch_size

for i in range(num_batches):
    tuner.search(data.train_x[batch_start:batch_end], data.train_y[batch_start:batch_end], epochs=10, validation_split=val_split, callbacks=[stop_early])
    batch_start = batch_end
    batch_end = batch_end*2
    if (data.train_x.shape[0]-batch_end)<batch_size:
        batch_end = -1

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')}, the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}, and 
the optimal dropout ration for the Dropout layers is {best_hps.get('dropout_ratio')}
""")

print(f"Units: {best_hps.get('units')}")

print(f"Learning rate: {best_hps.get('learning_rate')}")

print(f"Dropout ratio: {best_hps.get('dropout_ratio')}")

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(data.train_x, data.train_y, epochs=50, validation_split=val_split)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

params = {}
params['epoch'] = best_epoch
params['batch_size'] = batch_size
params['val_split'] = val_split
params['test_size'] = test_size
params['embedding_multiplier'] = embedding_dim_multiplier
params["units"] = best_hps.get('units')
params["lr"] = best_hps.get('learning_rate')
params["dropout_ratio"] = best_hps.get('dropout_ratio')
params['max_seq_len'] = max_seq_len
hp_values = json.dumps(params)
with open('params.txt', 'w') as hp_file:
    hp_file.write(hp_values)

# Reinstate the model with the best epoch
hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(data.train_x, data.train_y, epochs=best_epoch, validation_split=val_split)

# Evaluate the hypermodel on the test data
eval_result = hypermodel.evaluate(data.test_x, data.test_y)
print("[test loss, test accuracy]:", eval_result)

