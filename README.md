# Intent engine
From GPU instance:
This is the test version of the NLP intent engine that was trained and built in Transformers based deep neural networks using tensorflow.
This is an alternative candidate that might replace the current Rasa based NLP intent engine once tested, peer reviewed and proven to have higher accuracy of classifying intents, more lightweight to run in containers, and most importantly more reliable and scalable.

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

  

