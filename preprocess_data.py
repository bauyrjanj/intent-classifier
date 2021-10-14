from datetime import datetime
import helpers

# Load data, tokenizer and prepare data
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("\n{} : Started loading the data, preparing the data and creating the tokenizer....\n".format(now))
raw_data = helpers.load_data()
classes = raw_data.intent.unique().tolist()
train, test = helpers.prepare_data(raw_data)
tokenizer = helpers.create_tokenizer()
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("{} : Data was loaded, tokenizer was created....\n".format(now))

# Data preprocessing
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("{} : Started data preprocessing....\n".format(now))
data = helpers.preproc_data(train, test, tokenizer, classes)
create_config_file = helpers.create_classes_file(classes)

max_seq_len = data.max_seq_len # Save this to params.txt

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("\n{} : Finished data preprocessing....\n".format(now))

