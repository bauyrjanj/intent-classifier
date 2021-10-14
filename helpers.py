import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from bert.tokenization.bert_tokenization import FullTokenizer
import model_config
import json
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from tensorflow import keras
import warnings

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

def load_data():
    data_file = model_config.data_file
    data = pd.read_json(data_file, orient='table')
    return data

def prepare_data(df):
    X = df['text']
    y = df['intent']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=model_config.test_size, random_state=42, stratify=y)

    frame = {'text': X_train, 'intent': y_train}
    train = pd.DataFrame(frame)

    frame = {'text': X_test, 'intent': y_test}
    test = pd.DataFrame(frame)
    return train, test

def create_tokenizer():
    tokenizer = FullTokenizer(vocab_file=os.path.join(model_config.bert_ckpt_dir, "vocab.txt"))
    return tokenizer

def create_classes_file(classes):
    classes = {"classes": classes}
    g = json.dumps(classes)
    with open('classes.txt', 'w') as classes_file:
        classes_file.write(g)

def load_classes_file():
    cls = open("classes.txt", "r")
    classes = cls.read()
    cls.close()

    classes = json.loads(classes)

    classes = classes["classes"]
    return classes

def visualize_data(data):
    _ = plt.figure(figsize=model_config.fig_size)
    _ = sns.countplot(data.intent, palette=HAPPY_COLORS_PALETTE)
    _ = plt.title("Number of texts per intent")
    _ = plt.xticks(rotation=model_config.xticks_rotation)
    plt.show()

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
            self.max_seq_len = max(self.max_seq_len, len(token_ids)*model_config.embedding_dim_multiplier)
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

def preproc_input_text(sentence, tokenizer, max_seq_len):
    pred_tokens = tokenizer.tokenize(sentence)
    pred_tokens = ["[CLS]"] + pred_tokens + ["[SEP]"]
    pred_token_ids = tokenizer.convert_tokens_to_ids(pred_tokens)
    pred_token_ids = pred_token_ids + [0] * (max_seq_len - len(pred_token_ids))
    pred_token_ids = np.array(pred_token_ids).reshape(1, -1)
    return pred_token_ids


def create_model(classes, max_seq_len):
    with tf.io.gfile.GFile(model_config.bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        bert = BertModelLayer.from_params(bert_params, name="bert")

    input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
    bert_output = bert(input_ids) # output: [batch_size, max_seq_len, hidden_size]

    print("bert shape", bert_output.shape)

    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    cls_out = keras.layers.Dropout(model_config.dropout_ratio)(cls_out)
    logits = keras.layers.Dense(units=model_config.dense_units, activation="relu")(cls_out)
    logits = keras.layers.Dropout(model_config.dropout_ratio)(logits)
    # logits = keras.layers.Dense(units=384, activation="relu")(logits)
    # logits = keras.layers.Dropout(model_config.dropout_ratio)(logits)
    logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)

    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))

    load_stock_weights(bert, model_config.bert_ckpt_file)
    return model

def view_model_summary(model):
    return model.summary()

def compile_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(model_config.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )

def train_model(train_x, train_y, model):
    warnings.filterwarnings("ignore")
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(x=train_x,
                        y=train_y,
                        validation_split=model_config.validation_split,
                        batch_size=model_config.batch_size,
                        shuffle=True,
                        epochs=model_config.epochs,
                        callbacks=[stop_early]
                        )
    return history, model


