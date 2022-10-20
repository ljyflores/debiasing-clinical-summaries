import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

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

# Read data, filter to only first note per patient
df = pd.read_csv('data/preprocessed.csv', lineterminator='\n')
df = df.sort_values(by=['SUBJECT_ID','HADM_ID','CHARTDATE'])\
                .groupby(['SUBJECT_ID','HADM_ID'])\
                .head(1)

df_temp = df[['TEXT',TARGET_VAR]].copy()
df_temp['text'] = df_temp['TEXT']

if args.mode=='classification':
    df_temp['label'] = df[TARGET_VAR].apply(lambda x: 1*(x=='BLACK'))
else:
    df_temp['label'] = df[TARGET_VAR].apply(float)
    
df_train, df_test = train_test_split(df_temp, test_size=0.2)

# Define evaluation metrics
def acc_balanced(pred, orig):
    idx_1, idx_0 = np.where(np.array(orig)==1)[0], np.where(np.array(orig)==0)[0]
    acc_1, acc_0 = np.mean(np.array(pred[idx_1])==1), np.mean(np.array(pred[idx_0])==0)
    print(f"Acc 1: {acc_1}, Acc 0: {acc_0}, Acc Avg: {np.mean([acc_1,acc_0])}")
    
def compute_metrics(p):
    pred, labels = p

    if args.mode=='regression':
        mse = mean_squared_error(pred, labels)
        result_dict = {"MSE": mse}
    else:
        enc = OneHotEncoder(handle_unknown='ignore')
        y_test_ohe = enc.fit_transform(np.array(labels).reshape(-1,1)).todense()

        accuracy = accuracy_score(y_true=labels, y_pred=np.argmax(pred, axis=1))
        acc_bal  = acc_balanced(np.argmax(pred, axis=1), labels)
        roc_auc  = roc_auc_score(y_test_ohe, pred)
        result_dict = {"Accuracy (Raw)": accuracy,
                       "Accuracy (Balanced)": acc_bal,
                       "ROC AUC": roc_auc}
    return result_dict


# Load tokenizer and model, tokenize data
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", 
                                                           num_labels=NUM_LABELS)
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

dataset_train = Dataset.from_pandas(df_train, split='train')
dataset_test  = Dataset.from_pandas(df_test,  split='test')

dataset_train = dataset_train.map(preprocess_function, batched=True)
dataset_test  = dataset_test.map(preprocess_function, batched=True)

dataset = DatasetDict()
dataset['train'] = dataset_train
dataset['test'] = dataset_test

dataset_train = dataset_train.remove_columns(['TEXT',TARGET_VAR,'text'])
dataset_test  = dataset_test.remove_columns(['TEXT',TARGET_VAR,'text'])

# Huggingface Trainer
training_args = TrainingArguments(
    output_dir=f"./results/{args.mode}",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()