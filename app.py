from flask import Flask, request, jsonify, url_for
from tensorflow import keras
import logging
import helpers
import model_config
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import numpy as np

API_PATH = '/api'
model_path = model_config.model_save_dir
tokenizer = helpers.create_tokenizer()
classes = helpers.load_classes_file()
max_seq_len = model_config.max_seq_len
model = keras.models.load_model(model_path)
model.summary()
sid = SentimentIntensityAnalyzer()
application = Flask(__name__)
log = logging.getLogger(__name__)

nlp = spacy.load("./output/model-last")

@application.route("{}/predict".format(API_PATH), methods=['POST'])
def predict():
    data = request.get_json(force=True)
    sentence = data['text']
    pred_token_ids = helpers.preproc_input_text(sentence, tokenizer, max_seq_len)
    predictions = model.predict(pred_token_ids)
    pred_index = predictions.argmax(axis=1)
    sentiment = sid.polarity_scores(sentence)
    compound_score = sentiment['compound']
    if compound_score>=0.05:
        sentiment = "POSITIVE"
    elif compound_score<=-0.05:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
    doc = nlp(sentence)
    result = dict()
    result["intent"] = classes[pred_index[0]]
    result["sentiment"] = sentiment
    result["entities"] = [{"entity": ent.label_, "value": ent.text} for ent in doc.ents]
    result["confidence"] = round(np.float64(predictions.max(axis=1)[0]),2)
    return jsonify(result)

@application.route("{}/health".format(API_PATH))
def health():
    log.info('%s is called' % url_for('health'))
    return 'ok'

if __name__=="__main__":
    application.run(host='0.0.0.0', port=5056)

