# -*- coding: utf-8 -*-
"""
# Train and Inference
"""

import torch
torch.cuda.is_available()

import pandas as pd                                     # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np                                      # linear algebra
import re                                               # regular expressions for pattern matching in text 
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold     # Stratified K-Folds cross-validator
skf = StratifiedKFold(n_splits=5)                       # 5-fold cross-validation

# change matplotlib parameters
plt.rcParams['figure.figsize'] = [15, 10]
plt.rcParams.update({'font.size': 30})

"""# Training

## Imports
"""

from collections import deque
import random
import copy
import os

import torch                                                                                # PyTorch library for deep learning models and Tensors manipulation 
import sklearn.metrics as metrics                                                           # Metrics for evaluating the model  
from sklearn.model_selection import train_test_split                                        # Split the data into training and testing sets
import torch.nn as nn                                                                       # Neural network modules and loss functions for training            
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler    # Dataset wrapping tensors and sampling
from torch.optim.lr_scheduler import ReduceLROnPlateau                                      # Reduce learning rate when a metric has stopped improving
from torch.optim import AdamW                                                               # Adam optimizer with weight decay

# BERT model and tokenizer for sequence classification 
from transformers import BertForPreTraining, BertModel, AutoTokenizer, BertForSequenceClassification, RobertaForSequenceClassification 

from tqdm.notebook import tqdm, trange                                                     # Instantly make your loops show a smart progress meter

#import emoji
from nltk.corpus import stopwords

random_seed = 0
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)

"""## Flags"""

normalize_test_flag = True

# fold
k=0

"""## Utils"""

def normalize_text(tweets):
    # Filter only the tweets that are of type string
    tweets = [text for text in tweets if isinstance(text, str)]

    normalized_tweets = []
    for text in tweets:
        text = text.replace('&amp;', '&')       # Replace HTML escape character
        text = text.replace('\xa0', '')         # Replace non-breaking space
        text = re.sub(r'http\S+', '', text)     # Remove URLs
        text = " ".join(text.split())           # Remove extra whitespaces
        normalized_tweets.append(text)          # Append to the list

    return normalized_tweets

"""## Load data"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

#data_path = '../data/covid-latent/'
#data_path = '../data/covid-latent/undersampling/'
#data_path = '../data/stance-detection-in-covid-19-tweets/stay_at_home_orders/' #face_masks, school_closures, stay_at_home_orders, fauci
data_path = '../data/russian-troll-tweets/'
#data_path = '../data/COVIDSenti/'
#data_path = '../data/birdwatch/'
#data_path = '../data/mediaeval22/old_task1/'

filelist = os.listdir(data_path)

# Take only the first two files
filelist_subset = filelist[:2]

# Read the first two files into dataframes
df_list = [pd.read_csv(data_path + file) for file in filelist_subset]

test_df = df_list[k]                                    # Take the k-th dataframe as the test set

train_df = pd.concat(df_list[:k]+df_list[k+1:])         # Take all the dataframes except the k-th as the training set
test_df = pd.concat([train_df, test_df])                # Concatenate the test set to the training set


#tw_train = train_df['tweet'].tolist()
#tw_test = test_df['tweet'].tolist()

tw_train = train_df['content']
tw_test = test_df['content']
# ids_test = test_df['tweet'].tolist()

print(tw_test)

if normalize_test_flag:
    tw_train = normalize_text(tw_train)         # Normalize the training tweets
    tw_test = normalize_text(tw_test)           # Normalize the test tweets

#emotion
#train_df['emotion'][train_df['emotion'].isna()]='N'
#labels_train = train_df['emotion'].to_numpy()
#labels_train[labels_train=='N']=0
#labels_train[labels_train=='H']=1
#labels_train[labels_train=='A']=2
#labels_train[labels_train=='S']=3
#labels_train[labels_train=='F']=4
#labels_train = labels_train.tolist()

#sentiment
#labels_train = train_df['label'].to_numpy()
#labels_train[labels_train=='neu']=1
#labels_train[labels_train=='pos']=2
#labels_train[labels_train=='neg']=0
#labels_train = labels_train.tolist()

#political bias
labels_train = train_df['account_category'].to_numpy()
labels_train[labels_train=="Unknown"]=1
labels_train[labels_train=="NonEnglish"]=1
labels_train[labels_train=="Commercial"]=1
labels_train[labels_train=="NewsFeed"]=1
labels_train[labels_train=="HashtagGamer"]=1
labels_train[labels_train=="Fearmonger"]=1
labels_train[labels_train=="LeftTroll"]=0
labels_train[labels_train=="RightTroll"]=2
labels_train = labels_train.tolist()

#emotion
#test_df['emotion'][test_df['emotion'].isna()]='N'
#labels_test = test_df['emotion'].to_numpy()
#labels_test[labels_test=='N']=0
#labels_test[labels_test=='H']=1
#labels_test[labels_test=='A']=2
#labels_test[labels_test=='S']=3
#labels_test[labels_test=='F']=4
#labels_test = labels_test.tolist()

#sentiment
# labels_test = test_df['label'].to_numpy()
# labels_test[labels_test=='neu']=1
# labels_test[labels_test=='pos']=2
# labels_test[labels_test=='neg']=0
# labels_test = labels_test.tolist()

#political bias
labels_test = test_df['account_category'].to_numpy()
labels_test[labels_test=="Unknown"]=1
labels_test[labels_test=="NonEnglish"]=1
labels_test[labels_test=="Commercial"]=1
labels_test[labels_test=="NewsFeed"]=1
labels_test[labels_test=="HashtagGamer"]=1
labels_test[labels_test=="Fearmonger"]=1
labels_test[labels_test=="LeftTroll"]=0
labels_test[labels_test=="RightTroll"]=2
labels_test = labels_test.tolist()

ids_test = [i for i in range(0, len(test_df))]

#labels_train = [[l-1 for l in L] for L in labels_train]
#labels_test = [[l-1 for l in L] for L in labels_test]

weights = [len(labels_train)/w for w in [labels_train.count(a) for a in range(0, 3)]]       # Calculate the weights for the classes
weights = torch.FloatTensor(weights).to(device)                                             # Convert the weights to a tensor and move to the device
weights

# Load the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert')      

# Tokenize the input
tokenized_input = tokenizer(tw_train)                                                                                                      

m = 0
for tokens in tokenized_input['input_ids']:             # Find the maximum number of tokens in the input                        
    if len(tokens)>m:                                                                       
        m=len(tokens)
m

# Set the maximum number of tokens for each input
MAX_LEN = 64

# Tokenize the data in smaller batches to save memory
def batch_tokenize(texts, tokenizer, batch_size=100, max_length=MAX_LEN):
    input_ids = []
    token_type_ids = []
    attention_masks = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]                 # Extract the current batch

        # Tokenize the batch                    
        tokenized_batch = tokenizer(batch_texts, max_length=max_length, padding='max_length', truncation=True)

        input_ids.extend(tokenized_batch['input_ids'])              # Append the input ids
        token_type_ids.extend(tokenized_batch['token_type_ids'])    # Append the token type ids
        attention_masks.extend(tokenized_batch['attention_mask'])   # Append the attention masks

        # Explicitly release the memory of the processed batch
        del tokenized_batch

    return input_ids, token_type_ids, attention_masks

# Tokenize the training and test data
train_input_ids, train_token_type_ids, train_attention_mask = batch_tokenize(tw_train, tokenizer)
test_input_ids, test_token_type_ids, test_attention_mask = batch_tokenize(tw_test, tokenizer)

# Convert the train input ids, token type ids, attention masks and labels to torch tensors
train_input_ids = torch.tensor(train_input_ids, dtype=torch.long)
train_token_type_ids = torch.tensor(train_token_type_ids, dtype=torch.long)
train_attention_mask = torch.tensor(train_attention_mask, dtype=torch.long)

# Convert the test input ids, token type ids, attention masks and labels to torch tensors
test_input_ids = torch.tensor(test_input_ids, dtype=torch.long)
test_token_type_ids = torch.tensor(test_token_type_ids, dtype=torch.long)
test_attention_mask = torch.tensor(test_attention_mask, dtype=torch.long)
test_ids = torch.tensor(ids_test, dtype=torch.long)

# Convert the train and test labels to torch tensors
train_labels = torch.tensor(labels_train, dtype=torch.long)
test_labels = torch.tensor(labels_test, dtype=torch.long)

batch_size = 8              # 16 for emotion and 8 for sentiment and political bias

# Create the train and test datasets
train_data = TensorDataset(train_input_ids, train_attention_mask, train_labels, train_token_type_ids)
test_data = TensorDataset(test_input_ids, test_attention_mask, test_labels, test_token_type_ids, test_ids)

# Create the train and test dataloaders
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

"""## Models"""

class CovidTwitterBertClassifier(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.bert = BertForPreTraining.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2') 
        self.bert.cls.seq_relationship = nn.Linear(1024, n_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, token_type_ids, input_mask): 
        outputs = self.bert(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = input_mask) 

        logits = outputs[1]

        return logits

model = CovidTwitterBertClassifier(3)                   # 5 for emotion and 3 for sentiment and political bias


model.to(device)

# Set the optimizer, scheduler and loss function
optimizer = AdamW(model.parameters(),
                  lr=1e-5,
                  #lr=3e-5,
                  weight_decay = 0.01)

scheduler = ReduceLROnPlateau(optimizer, patience=4, factor=0.3)

criterion = nn.CrossEntropyLoss(weight = weights)

"""## Training loop"""
epochs = 15

best_F1 = 0
best_ACC = 0
best_loss = 999
best_acc = 0
best_state_dict = model.state_dict()
best_epoch = 0

for e in trange(0, epochs, position=0, leave=True):

    # Training
    print('Starting epoch ', e)
    model.train()

    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0                                  # Number of training examples and steps

    for step, batch in enumerate(tqdm(train_dataloader)):

        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from the dataloader
        b_input_ids, b_input_mask, b_labels, b_token_type_ids = batch

        # Clear previously calculated gradients
        b_labels = b_labels.float()
        optimizer.zero_grad()

        # Forward pass
        logits = model(b_input_ids, b_token_type_ids, b_input_mask)

        # Calculate the loss
        loss = criterion(logits, b_labels.long())
        loss.backward()
        optimizer.step()

        # Update the training loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss/nb_tr_steps))


    # eval

    logits_full = []
    ground_truth_full = []

    model.eval()
    eval_loss = 0
    steps=0
    for step, batch in enumerate(tqdm(test_dataloader)):

        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from the dataloader
        b_input_ids, b_input_mask, b_labels, b_token_type_ids, b_ids = batch

        b_labels = b_labels.float()

        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():

            # Forward pass, calculate logit predictions
            logits = model(b_input_ids, b_token_type_ids, b_input_mask)
            loss = criterion(logits, b_labels.long())


        # Move logits and labels to CPU
        logits = logits.detach().cpu().tolist()
        logits_full.extend(logits)
        ground_truth = b_labels.detach().cpu().tolist()
        ground_truth_full.extend(ground_truth)

        steps+=1
        eval_loss+=loss.detach().item()     # Update the evaluation loss


    scheduler.step(eval_loss/steps)
    LOSS = eval_loss/steps
    F1 = metrics.f1_score(np.array(logits_full).argmax(axis=1), np.array(ground_truth_full), average='micro')
    ACC = metrics.accuracy_score(np.array(logits_full).argmax(axis=1), np.array(ground_truth_full))

    if F1> best_F1:
        best_loss = LOSS
        best_F1 = F1
        best_ACC = ACC
        best_state_dict = copy.deepcopy(model.state_dict())
        best_epoch = e

    print("\t Eval loss: {}".format(LOSS))
    print("\t Eval F1: {}".format(F1))
    print("\t Eval ACC: {}".format(ACC))
    print("---"*25)
    print("\n")

print("Best epoch", best_epoch)
print("\t Eval loss: {}".format(best_loss))
print("\t Eval F1: {}".format(best_F1))
print("---"*25)
print("\n")

# Save the best model
torch.save(best_state_dict, '../data/models/emotion_undersampling_CV'+str(k)+'_e'+str(best_epoch)+'_'+str(round(best_F1, 3))+'.pth')

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType
from tqdm import tqdm, trange

os.environ["TOKENIZERS_PARALLELISM"] = "false"

spark = SparkSession.builder.appName("COVIDTweetStream").getOrCreate()

schema = StructType([
    StructField("content", StringType(), True),
    StructField("account_category", StringType(), True)
])

def normalize_text(texts):
    return texts

def process_data(df):

    train_df = df.sample(fraction=0.8, seed=42)     # 80% of the data for training
    test_df = df.subtract(train_df)                 # 20% of the data for testing

    # Normalize the tweets
    tw_train = train_df.select('content').rdd.flatMap(lambda x: x).collect()
    tw_test = test_df.select('content').rdd.flatMap(lambda x: x).collect()

    if normalize_test_flag:
        tw_train = normalize_text(tw_train)
        tw_test = normalize_text(tw_test)

    labels_train = train_df.select('account_category').rdd.flatMap(lambda x: x).collect()
    labels_train = [1 if cat in ["Unknown", "NonEnglish", "Commercial", "NewsFeed", "HashtagGamer", "Fearmonger"] else 0 if cat == "LeftTroll" else 2 for cat in labels_train]

    labels_test = test_df.select('account_category').rdd.flatMap(lambda x: x).collect()
    labels_test = [1 if cat in ["Unknown", "NonEnglish", "Commercial", "NewsFeed", "HashtagGamer", "Fearmonger"] else 0 if cat == "LeftTroll" else 2 for cat in labels_test]

    ids_test = [i for i in range(len(test_df.collect()))]

    model.load_state_dict(torch.load('../data/models/russian0_e5_0.681.pth'))
    model.eval()

    logits_full = []
    ground_truth_full = []
    ids_full = []

    eval_loss = 0
    steps = 0

    for step, batch in enumerate(tqdm(test_dataloader)):
        batch = tuple(t.to(device) for t in batch)                                  # Add batch to GPU
        b_input_ids, b_input_mask, b_labels, b_token_type_ids, test_ids = batch     # Unpack the inputs from the dataloader
        b_labels = b_labels.float()                                                 # Convert the labels to float

        with torch.no_grad():
            logits = model(b_input_ids, b_token_type_ids, b_input_mask)            # Forward pass, calculate logit predictions

        logits = logits.detach().cpu().tolist()
        logits_full.extend(logits)
        ground_truth = b_labels.detach().cpu().tolist()
        ground_truth_full.extend(ground_truth)              
        ids = test_ids.detach().cpu().tolist()
        ids_full.extend(ids)
        steps += 1

    scheduler.step(eval_loss / steps)
    LOSS = eval_loss / steps
    F1 = metrics.f1_score(np.array(logits_full).argmax(axis=1), np.array(ground_truth_full), average='micro')
    ACC = metrics.accuracy_score(np.array(logits_full).argmax(axis=1), np.array(ground_truth_full))

    print("\t Eval loss: {}".format(LOSS))
    print("\t Eval F1: {}".format(F1))
    print("\t Eval ACC: {}".format(ACC))
    print("---" * 25)
    print("\n")

    df_result = pd.DataFrame()
    df_result['ids'] = ids_full
    df_result['political_bias'] = np.array(logits_full).argmax(axis=1).tolist()
    df_result.to_csv(data_path + 'emotion_full.csv', index=False)

    A = np.zeros((3, 3))

    test_account_categories = test_df.select('account_category').rdd.flatMap(lambda x: x).collect()
    political_biases = df_result.sort_values(by='ids')['political_bias'].tolist()

    for i in trange(0, len(df_result)):
        A[test_account_categories[i]][political_biases[i]] += 1

    for i in range(0, 3):
        A[i, :] = A[i, :] / A[i, :].sum()

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Create a heatmap
    sns.heatmap(A, cmap=sns.light_palette("darkred", as_cmap=True), yticklabels=['Left', 'Other', 'Right'], xticklabels=['Positive', "Neutral", "Negative"])
    plt.savefig('../heatmap.png')

# Create a DataFrame representing the stream of input text
df = spark \
    .readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

json_df = df.select(from_json(col("value"), schema).alias("data")).select("data.*")

query = json_df.writeStream.foreachBatch(lambda batch_df, batch_id: process_data(batch_df)).start()

query.awaitTermination()
