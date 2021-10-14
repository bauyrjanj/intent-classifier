import pandas as pd
import os
data_file = 'nlu_data_cleaned.csv'
df = pd.read_csv(data_file, encoding='cp1252')
os.chdir("../")
path = 'nlu.json'
json_format_df = df.to_json(path, orient='table')

