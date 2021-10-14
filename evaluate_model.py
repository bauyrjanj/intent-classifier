from datetime import datetime
import os
import warnings
import pickle
import json
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import preprocess_data
import helpers
import model_config

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("\n{}: Started loading the model...........\n".format(now))
model_path = model_config.model_save_dir
model = keras.models.load_model(model_path)
hist = pickle.load(open('trainHistoryDict','rb'))
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("\n{}: Successfully loaded the model\n".format(now))

data = preprocess_data.data
classes = helpers.load_classes_file()

os.chdir("results")

# Model evaluation metrics
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("\n{}: Calculating the train and test accuracy of the model...........\n".format(now))
_, train_acc = model.evaluate(data.train_x, data.train_y)
_, test_acc = model.evaluate(data.test_x, data.test_y)
print("train acc: ", train_acc)
print("test acc: ", test_acc)
accuracy = {}
accuracy['test'] = test_acc
accuracy['train'] = train_acc
with open('accuracy.json', 'w') as f:
    json.dump(accuracy, f)

# Classification report
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("{}: Constructing a detailed classification report....\n".format(now))
warnings.filterwarnings("ignore")
y_pred = model.predict(data.test_x).argmax(axis=-1)
report = classification_report(data.test_y, y_pred, target_names=classes, output_dict=True)
with open('report_intent.json', 'w') as f:
    json.dump(report, f)
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("{}: Successfully saved the report....\n".format(now))

# Confusion matrix
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("{}: Creating confusion matrix..........\n".format(now))
cm = confusion_matrix(data.test_y, y_pred)
df_cm = pd.DataFrame(cm, index=classes, columns=classes)
plt.figure(figsize=(17,9))
hmap = sns.heatmap(df_cm, annot=True, fmt="d")
hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=90, ha='right')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig("confusion_matrix.png", bbox_inches='tight')
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("{}: Confusion matrix was saved in the directory "
      "{}".format(now, os.path.dirname(os.path.realpath("bert_intent_classifier"))))
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Graphs for accuracy and loss
print("\n{}: Constructing the graph of loss over training epochs......\n".format(now))
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.plot(hist['loss'])
ax.plot(hist['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.title('Loss over training epochs')
plt.savefig("loss_training_epochs.png", bbox_inches='tight')
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("{}: The graph was saved in the directory "
      "{}".format(now, os.path.dirname(os.path.realpath("bert_intent_classifier"))))

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("\n{}: Constructing the graph of accuracy over training epochs......\n".format(now))
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.plot(hist['acc'])
ax.plot(hist['val_acc'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.title('Accuracy over training epochs')
plt.savefig("accuracy_training_epochs.png", bbox_inches='tight')
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("{}: The graph was saved in the directory "
      "{}".format(now, os.path.dirname(os.path.realpath("bert_intent_classifier"))))

os.chdir("../")