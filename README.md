# Intent engine
This is the test version of the NLP intent engine. Pre-trained BERT-base-uncased model as used from Hugging Face and was further trained using Tensorflow 2.4.0.

# Network architecture

1. Layer: bert-base-uncased: [.,768] 
2. Layer: Lambda: [.,768]
3. Layer: Dense-> ReLu:[.,768]
4. Layer: Dropout: [.,384]
5. Layer: Dense-> ReLu: [.,2]

# What does the code in this repo do
The code in this repo in particular performs model training, hyperparameter tuning, model evaluation, inference and saves the ready-to-deploy model with its parameters and configurations. 

# Set up environment 
* Requirement: ```anaconda```, ```pip```, ```python 3```
* It is a good practice to create a new conda environment and here is how to create one:
  ```
  conda create -n "name_of_new_environment" python==3.8.5
  conda activate "name_of_new_environment"
  ```
* Install dependencies (use Anaconda Prompt): 
  ```
  pip install --upgrade pip 
  pip install -r requirements.txt 
  ```
  
# How to train the model

Ensure your current directory has all the files from this repo.
```
python train_model.py

```

# How to perform hyperparameter tuning

Ensure your current directory has all the files from this repo. This step will take a longer time depending on the 
hyperparameter search space.
```
python hyperparameter-tuner.py

```

# How to do inference

Ensure your current directory has all the files from this repo.
```
python run_inference.py

```

  

