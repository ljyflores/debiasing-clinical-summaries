import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (DataCollatorWithPadding,
                          T5ForConditionalGeneration,
                          T5Tokenizer,
                          TrainingArguments, 
                          Seq2SeqTrainer,
                          Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq)
from nltk import sent_tokenize
from sklearn.model_selection import train_test_split
import evaluate

rouge_score = evaluate.load("rouge")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small",
                                            output_scores=True,
                                            output_hidden_states=True)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

max_input_length = 512
max_target_length = 512

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    return rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["TEXT"],
        max_length=max_input_length,
        truncation=True,
        padding=True
    )
    model_inputs["labels"]                 = model_inputs["input_ids"]
    model_inputs["decoder_input_ids"]      = model_inputs["input_ids"]
    model_inputs["decoder_attention_mask"] = model_inputs["attention_mask"]
    return model_inputs

df = pd.read_csv('data/preprocessed.csv', lineterminator='\n')
df = df[['TEXT','ETHNICITY']]
df_train, df_test = train_test_split(df, test_size=0.2)

dataset_train = Dataset.from_pandas(df_train, split='train')
dataset_test  = Dataset.from_pandas(df_test,  split='test')

dataset_train = dataset_train.map(preprocess_function, batched=True)
dataset_test  = dataset_test.map(preprocess_function, batched=True)

dataset = DatasetDict()
dataset['train'] = dataset_train
dataset['test'] = dataset_test

dataset['train'] = dataset['train'].remove_columns(['TEXT',
                                 'ETHNICITY', 
                                 '__index_level_0__'])
dataset['test'] = dataset['test'].remove_columns(['TEXT',
                                 'ETHNICITY', 
                                 '__index_level_0__'])

batch_size = 8
num_train_epochs = 8
# Show the training loss with every epoch

args = Seq2SeqTrainingArguments(
    output_dir=f"results/generator",
    evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()