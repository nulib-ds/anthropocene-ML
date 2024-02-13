import pandas as pd
import numpy as np
import math
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from transformers import TFDistilBertModel, DistilBertTokenizerFast, DistilBertConfig


from tensorflow.keras.layers import (
    Input,
    Dense,
    Embedding,
    Flatten,
    Dropout,
    GlobalMaxPooling1D,
    GRU,
    concatenate,
)

def print_metric(model, x_train, y_train, x_val, y_val):
    train_acc = dict(model.evaluate(x_train, y_train, verbose=0, return_dict=True))['accuracy']
    val_acc = dict(model.evaluate(x_val, y_val, verbose=0, return_dict=True))['accuracy']

    val_preds = model.predict(x_val)
    val_preds = val_preds > 0.5

    print("")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Val accuracy: {val_acc:.4f}")
    print("")
    print(f"Validation f1 score: {sklearn.metrics.f1_score(val_preds_bool, y_val)}")

# Using DistilBERT:
model_class, tokenizer_class, pretrained_weights = (TFDistilBertModel, DistilBertTokenizerFast, 'distilbert-base-uncased')

pretrained_bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

def get_pretrained_bert_model(config=pretrained_weights):
    if not config:
        config = DistilBertConfig(num_labels=2)

    return model_class.from_pretrained(pretrained_weights, config=config)

# Split the data into training and validation sets

dataset = pd.read_csv("Emails.csv")

train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42)

print(train_df, test_df)




print(train_df.info())

print("")
print("train rows:", len(train_df.index))
print("test rows:", len(test_df.index))



print("label counts:")
train_df.target.value_counts()

class EmailPreProcessor:
    :

