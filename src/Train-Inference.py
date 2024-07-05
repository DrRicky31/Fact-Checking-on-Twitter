# -*- coding: utf-8 -*-
"""Train-Inference.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1sdhX_mYJ2Z6ssYBXhpLs0lrkT49IUT3D

# Train and Inference
"""

import torch
torch.cuda.is_available()

import pandas as pd
import numpy as np
import re
import json
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)

# change matplotlib parameters
plt.rcParams['figure.figsize'] = [15, 10]
plt.rcParams.update({'font.size': 30})

"""# Training

## Imports
"""

import numpy as np
import pandas as pd
from collections import deque
import random
import copy
import os

import pandas as pd
import numpy as np
import re
import json
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForPreTraining, BertModel, AutoTokenizer, BertForSequenceClassification, RobertaForSequenceClassification
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW

from tqdm.notebook import tqdm, trange

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

import re

def normalize_text(tweets):
    # Filtra solo i tweet che sono di tipo stringa
    tweets = [text for text in tweets if isinstance(text, str)]

    normalized_tweets = []
    for text in tweets:
        text = text.replace('&amp;', '&')
        text = text.replace('\xa0', '')
        text = re.sub(r'http\S+', '', text)
        text = " ".join(text.split())
        normalized_tweets.append(text)

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

# Prendi solo i primi tre file
filelist_subset = filelist[:2]

# Leggi i primi tre file in dataframe
df_list = [pd.read_csv(data_path + file) for file in filelist_subset]

test_df = df_list[k]

train_df = pd.concat(df_list[:k]+df_list[k+1:])
test_df = pd.concat([train_df, test_df])


#tw_train = train_df['tweet'].tolist()
#tw_test = test_df['tweet'].tolist()

tw_train = train_df['content']
tw_test = test_df['content']
# ids_test = test_df['tweet'].tolist()

print(tw_test)

if normalize_test_flag:
    tw_train = normalize_text(tw_train)
    tw_test = normalize_text(tw_test)

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

weights = [len(labels_train)/w for w in [labels_train.count(a) for a in range(0, 3)]]
weights = torch.FloatTensor(weights).to(device)
weights

tokenizer = AutoTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert')

tokenized_input = tokenizer(tw_train)

m = 0
for tokens in tokenized_input['input_ids']:
    if len(tokens)>m:
        m=len(tokens)
m

# Imposta il massimo numero di token per ogni input
MAX_LEN = 64

# Tokenizza i dati in batch più piccoli per risparmiare memoria
def batch_tokenize(texts, tokenizer, batch_size=100, max_length=MAX_LEN):
    input_ids = []
    token_type_ids = []
    attention_masks = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        tokenized_batch = tokenizer(batch_texts, max_length=max_length, padding='max_length', truncation=True)

        input_ids.extend(tokenized_batch['input_ids'])
        token_type_ids.extend(tokenized_batch['token_type_ids'])
        attention_masks.extend(tokenized_batch['attention_mask'])

        # Rilascia esplicitamente la memoria del batch processato
        del tokenized_batch

    return input_ids, token_type_ids, attention_masks

# Tokenizza i dati di addestramento e di test in batch
train_input_ids, train_token_type_ids, train_attention_mask = batch_tokenize(tw_train, tokenizer)
test_input_ids, test_token_type_ids, test_attention_mask = batch_tokenize(tw_test, tokenizer)

# Converti le liste in tensori torch
train_input_ids = torch.tensor(train_input_ids, dtype=torch.long)
train_token_type_ids = torch.tensor(train_token_type_ids, dtype=torch.long)
train_attention_mask = torch.tensor(train_attention_mask, dtype=torch.long)

test_input_ids = torch.tensor(test_input_ids, dtype=torch.long)
test_token_type_ids = torch.tensor(test_token_type_ids, dtype=torch.long)
test_attention_mask = torch.tensor(test_attention_mask, dtype=torch.long)
test_ids = torch.tensor(ids_test, dtype=torch.long)

# Converti le etichette in tensori torch (assumendo che le etichette siano interi)
train_labels = torch.tensor(labels_train, dtype=torch.long)
test_labels = torch.tensor(labels_test, dtype=torch.long)

batch_size = 8 #

train_data = TensorDataset(train_input_ids, train_attention_mask, train_labels, train_token_type_ids)
test_data = TensorDataset(test_input_ids, test_attention_mask, test_labels, test_token_type_ids, test_ids)


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

model = CovidTwitterBertClassifier(3) # 5 for emotion and 3 for sentiment and political bias


model.to(device)

#optimizer_grouped_parameters
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
    nb_tr_examples, nb_tr_steps = 0, 0

    for step, batch in enumerate(tqdm(train_dataloader)):

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels, b_token_type_ids = batch

        b_labels = b_labels.float()
        optimizer.zero_grad()

        logits = model(b_input_ids, b_token_type_ids, b_input_mask)

        loss = criterion(logits, b_labels.long())
        loss.backward()
        optimizer.step()

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

        b_input_ids, b_input_mask, b_labels, b_token_type_ids, b_ids = batch

        b_labels = b_labels.float()

        with torch.no_grad():

            logits = model(b_input_ids, b_token_type_ids, b_input_mask)
            loss = criterion(logits, b_labels.long())



        logits = logits.detach().cpu().tolist()
        logits_full.extend(logits)
        ground_truth = b_labels.detach().cpu().tolist()
        ground_truth_full.extend(ground_truth)

        steps+=1
        eval_loss+=loss.detach().item()


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

torch.save(best_state_dict, '../data/covid-latent/models/emotion_undersampling_CV'+str(k)+'_e'+str(best_epoch)+'_'+str(round(best_F1, 3))+'.pth')

"""# Inference

## Load Data
"""

#data_path = '../data/covid-latent/'
#data_path = '../data/covid-latent/undersampling/'
#data_path = '../data/stance-detection-in-covid-19-tweets/stay_at_home_orders/' #face_masks, school_closures, stay_at_home_orders, fauci
#data_path = '../data/russian-troll-tweets/'
data_path = '../data/COVIDSenti/'
#data_path = '../data/birdwatch/'
#data_path = '../data/mediaeval22/old_task1/'

filelist = os.listdir(data_path)

# Prendi solo i primi tre file
filelist_subset = filelist[:2]

df_list = [pd.read_csv(data_path+file) for file in filelist_subset]


test_df = df_list[k]

train_df = pd.concat(df_list[:k]+df_list[k+1:])
test_df = pd.concat([train_df, test_df])


tw_train = train_df['content']
tw_test = test_df['content']
#ids_test = test_df['tweet'].tolist()


if normalize_test_flag:
    tw_train = normalize_text(tw_train)
    tw_test = normalize_text(tw_test)

#emotion
train_df['emotion'][train_df['emotion'].isna()]='N'
labels_train = train_df['emotion'].to_numpy()
labels_train[labels_train=='N']=0
labels_train[labels_train=='H']=1
labels_train[labels_train=='A']=2
labels_train[labels_train=='S']=3
labels_train[labels_train=='F']=4
labels_train = labels_train.tolist()

#sentiment
labels_train = train_df['label'].to_numpy()
labels_train[labels_train=='neu']=1
labels_train[labels_train=='pos']=2
labels_train[labels_train=='neg']=0
labels_train = labels_train.tolist()

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

#stance
labels_train = train_df['Stance'].to_numpy()
labels_train[labels_train=="FAVOR"]=2
labels_train[labels_train=="NONE"]=1
labels_train[labels_train=="AGAINST"]=0
labels_train = labels_train.tolist()

#veracity
labels_train = train_df['note'].to_numpy()
labels_train[labels_train=="MISINFORMED_OR_POTENTIALLY_MISLEADING"]=0
labels_train[labels_train=="NOT_MISLEADING"]=1
labels_train = labels_train.tolist()

#conspiracy
labels_train = train_df['conspiracy'].tolist()

#emotion
test_df['emotion'][test_df['emotion'].isna()]='N'
labels_test = test_df['emotion'].to_numpy()
labels_test[labels_test=='N']=0
labels_test[labels_test=='H']=1
labels_test[labels_test=='A']=2
labels_test[labels_test=='S']=3
labels_test[labels_test=='F']=4
labels_test = labels_test.tolist()

#sentiment
labels_test = test_df['label'].to_numpy()
labels_test[labels_test=='neu']=1
labels_test[labels_test=='pos']=2
labels_test[labels_test=='neg']=0
labels_test = labels_test.tolist()

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

#stance
labels_test = test_df['Stance'].to_numpy()
labels_test[labels_test=="FAVOR"]=2
labels_test[labels_test=="NONE"]=1
labels_test[labels_test=="AGAINST"]=0
labels_test = labels_test.tolist()

#veracity
labels_test = test_df['note'].to_numpy()
labels_test[labels_test=="MISINFORMED_OR_POTENTIALLY_MISLEADING"]=0
labels_test[labels_test=="NOT_MISLEADING"]=1
labels_test = labels_test.tolist()

#conspiracy
labels_test = test_df['conspiracy'].tolist()


ids_test = [i for i in range(0, len(test_df))]

#labels_train = [[l-1 for l in L] for L in labels_train]
#labels_test = [[l-1 for l in L] for L in labels_test]

"""## Load model"""

model.load_state_dict(torch.load('../data/covid-latent/models/emotion_undersampling_CV0_e2_0.622.pth'))
model.eval()

logits_full = []
ground_truth_full = []
ids_full = []

eval_loss = 0
steps=0
for step, batch in enumerate(tqdm(test_dataloader)):

    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)

    b_input_ids, b_input_mask, b_labels, b_token_type_ids, test_ids = batch

    b_labels = b_labels.float()

    with torch.no_grad():

        logits = model(b_input_ids, b_token_type_ids, b_input_mask)
        #loss = criterion(logits, b_labels.long())



    logits = logits.detach().cpu().tolist()
    logits_full.extend(logits)
    ground_truth = b_labels.detach().cpu().tolist()
    ground_truth_full.extend(ground_truth)
    ids = test_ids.detach().cpu().tolist()
    ids_full.extend(ids)
    steps+=1
    #eval_loss+=loss.detach().item()

scheduler.step(eval_loss/steps)
LOSS = eval_loss/steps
F1 = metrics.f1_score(np.array(logits_full).argmax(axis=1), np.array(ground_truth_full), average='micro')
ACC = metrics.accuracy_score(np.array(logits_full).argmax(axis=1), np.array(ground_truth_full))


print("\t Eval loss: {}".format(LOSS))
print("\t Eval F1: {}".format(F1))
print("\t Eval ACC: {}".format(ACC))
print("---"*25)
print("\n")

df = pd.DataFrame()

df['ids'] = ids_full
df['emotion'] = np.array(logits_full).argmax(axis=1).tolist()
#df.to_csv(data_path+'masks'+str(k)+'.csv', index=False)
df.to_csv(data_path+'emotion_full.csv', index=False)

"""# Visu"""

A = np.zeros((3, 5))

for i in trange(0, len(df)):
    A[test_df['conspiracy'].tolist()[i]][df.sort_values(by='ids')['emotion'].tolist()[i]]+=1
for i in range(0, 3):
    A[i,:] = A[i,:]/A[i,:].sum()
A

#NHASF
#sns.light_palette("seagreen", as_cmap=True)
sns.heatmap(A, cmap = sns.light_palette("darkred", as_cmap=True), yticklabels=['No Conspiracy', 'Discussing', 'Promoting'], xticklabels=['N', 'H', 'A', 'S', 'F'])