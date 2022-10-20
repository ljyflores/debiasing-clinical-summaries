import pandas as pd
import numpy as np
from transformers import (AutoModelForSequenceClassification, 
                          DataCollatorWithPadding,
                          AutoTokenizer)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, mean_squared_error
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from utils import mse_balanced, acc_balanced
import argparse

# Parse argument for regression or classification
parser = argparse.ArgumentParser()
parser.add_argument('--mode',   type=str,   required=True)
args = parser.parse_args()

if args.mode not in ['classification','regression']:
    print("Mode must be either 'classification' or 'regression'")
    assert False

# Set variables for tasks
NUM_LABELS = 2 if args.mode == 'classification' else 1
TARGET_VAR = 'ETHNICITY' if args.mode == 'classification' else 'apsiii'
print(args.mode, NUM_LABELS, TARGET_VAR)

device = torch.device("cuda")

### UTILITY FUNCTIONS ###
def encode(examples, tokenizer):
    inputs = examples['text']
    targets = examples['label']
    
    tokenized_inputs = tokenizer(inputs, 
                                 return_tensors='pt', 
                                 padding=True
                                 )
    model_inputs = {}
    model_inputs['input_ids']      = tokenized_inputs['input_ids'][:,:512]
    model_inputs['attention_mask'] = tokenized_inputs['attention_mask'][:,:512]
    model_inputs['labels']         = targets

    return model_inputs

def evaluate(model, dataloader, tokenizer):
    result = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = encode(batch, tokenizer)
            output = model(input_ids      = inputs['input_ids'].to(device),
                           attention_mask = inputs['attention_mask'].to(device),
                           labels         = inputs['labels'].to(device))
            result.append(output.logits.cpu())
    return result


# Load tokenizer and model, tokenize data
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(f"results/{args.mode}/checkpoint-8500", 
                                                           num_labels=NUM_LABELS).to(device)

# Read data and split into train/test
df = pd.read_csv('data/preprocessed.csv', lineterminator='\n')
df_temp = df.sort_values(by=['SUBJECT_ID','HADM_ID','CHARTDATE'])\
                .groupby(['SUBJECT_ID','HADM_ID'])\
                .head(1)

if args.mode=='classification':
    df_temp = df_temp[['TEXT',TARGET_VAR]].copy()
    df_temp['label'] = df[TARGET_VAR].apply(lambda x: 1*(x=='BLACK'))
    df_temp['text'] = df_temp['TEXT']
else:
    df_temp = df_temp[['TEXT',TARGET_VAR,'ETHNICITY']].copy()
    df_temp['label'] = df[TARGET_VAR].apply(float)
    df_temp['text'] = df_temp['TEXT']

df_train, df_test, r_train, r_test = train_test_split(df_temp, 
                                                      1*(df_temp['ETHNICITY']=='BLACK'), 
                                                      test_size=0.2)

# Create datasets and evaluate
dataset_test  = Dataset.from_pandas(df_test,  split='test')
dataloader_test = DataLoader(dataset_test, batch_size=16, shuffle=False)

results = evaluate(model, dataloader_test, tokenizer)
results = torch.vstack(results).numpy()

if args.mode=='classification':
    enc = OneHotEncoder(handle_unknown='ignore')
    y_test_ohe = enc.fit_transform(np.array(df_test['label']).reshape(-1,1)).todense()
    acc_balanced(np.argmax(results, axis=1), df_test['label'])
    print(roc_auc_score(y_test_ohe, results))
else:
    pred = results.reshape(-1)
    orig = np.array(df_test['label'])
    race = np.array(r_test)
    mse_balanced(pred, orig, race)
    print(f"Test MSE: {mean_squared_error(pred, orig)}")