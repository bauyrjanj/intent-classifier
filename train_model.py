import warnings
from datetime import datetime
import pickle
import model_config
import helpers
import preprocess_data

data = preprocess_data.data
classes = helpers.load_classes_file()

# Model creation and training
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("{}: Started creating the model....\n".format(now))
model = helpers.create_model(classes, data.max_seq_len)
print("\nArchitecture of the model:")
helpers.view_model_summary(model)
helpers.compile_model(model)
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("{}: The model has been created and compiled successfully....\n".format(now))

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("{}: Started training the deep neural network model, this will take some time so feel free to grab a cup of coffee....\n".format(now))
history, model = helpers.train_model(data.train_x, data.train_y, model)
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("{}: Finished training the deep neural network model....\n".format(now))

# Save the new model
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("{}: Saving the newly trained model....\n".format(now))
warnings.filterwarnings("ignore")
model_path = model_config.model_save_dir
model.save(model_path)
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("{}: New model has been saved in the directory {}".format(now, model_path))

# Save the history
with open('trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

