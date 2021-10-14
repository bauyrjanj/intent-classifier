import model_config
from tensorflow import keras
import warnings
from datetime import datetime
import helpers
import numpy as np

# Load the model
warnings.filterwarnings("ignore")
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("\n{}: Started loading the model....\n".format(now))
model = keras.models.load_model(model_config.model_save_dir)

classes = helpers.load_classes_file()
max_seq_len = model_config.max_seq_len

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("{}: The model was successfully loaded\n".format(now))

# Create a tokenizer
warnings.filterwarnings("ignore")
tokenizer = helpers.create_tokenizer()
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("{}: The tokenizer was successfully created\n".format(now))

# Get the text from the user
text = input("\nPlease enter a sample text: ")

# Preprocess the input text
warnings.filterwarnings("ignore")
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("{}: Started preprocessing the input text\n".format(now))
pred_token_ids = helpers.preproc_input_text(text, tokenizer, max_seq_len)
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("{}: Finished preprocessing the input text\n".format(now))

# make prediction
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("{}: Started the inference process\n".format(now))
predictions = model.predict(pred_token_ids).argmax(axis=-1)
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("{}: Finished the inference process\n".format(now))

print("Predicted intent: {}".format(classes[predictions[0]]))

# predictions = model.predict(pred_token_ids)
# print("Raw predictions: {}".format(predictions))
# print("Shape of raw  predictions: {}".format(predictions.shape))
# print("Argmax: {}".format(predictions.argmax(axis=1)))
# print("Prediction score: {}".format(predictions.max(axis=1)))
# print("Type of prediction score: {}".format(type(predictions.max(axis=1)[0])))
# print("Type of prediction score: {}".format(np.float64(predictions.max(axis=1)[0])))



