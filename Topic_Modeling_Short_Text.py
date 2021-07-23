# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from preprocessing import tokenize, export_to_csv
from gsdmm import MovieGroupProcess
from topic_allocation import top_words, topic_attribution
from visualisation import plot_topic_notebook, save_topic_html
from sklearn.datasets import fetch_20newsgroups

import pickle
import matplotlib as plt
import pandas as pd
import numpy as np
import ast

# %% [markdown]
# # Topic Modeling on 20NewsGroups
# https://towardsdatascience.com/short-text-topic-modeling-70e50a57c883
# %% [markdown]
# ## Data selection

# %%
cats = ['talk.politics.mideast', 'comp.windows.x', 'sci.space']

newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=cats)
newsgroups_train_subject = fetch_20newsgroups(subset='train', categories=cats)

data = newsgroups_train.data
data_subject = newsgroups_train_subject.data

targets = newsgroups_train.target.tolist()
target_names = newsgroups_train.target_names

# %%
# Let's see if our topics are evenly distributed
df_targets = pd.DataFrame({'targets': targets})
order_list = df_targets.targets.value_counts()
order_list


# %% jupyter={"outputs_hidden": true}
def extract_first_sentence(data_subject):
    list_first_sentence = []
    for text in data:
        first_sentence = text.split(".")[0].replace("\n", "")
        list_first_sentence.append(first_sentence)
    return list_first_sentence


def extract_subject(data):
    c = 0
    s = "Subject:"
    list_subjects = []
    for new in data_subject:    
        lines = new.split("\n")
        b = 0 # loop out at the first "Subject:", they may be several and we want first one only
        for line in lines:
            if s in line and b == 0:
                subject = " ".join(line.split(":")[1:]).strip()
                subject = subject.replace('Re', '').strip()
                list_subjects.append(subject)
                c += 1
                b = 1
    return list_subjects
   
    
def concatenate(list_first_sentence, list_subjects):
    list_docs = []
    for i in range(len(list_first_sentence)):
        list_docs.append(list_subjects[i] + " " + list_first_sentence[i])
    return list_docs


list_first_sentence = extract_first_sentence(data)
list_subjects = extract_subject(data_subject)
list_docs = concatenate(list_first_sentence, list_subjects)


# %%
df = pd.DataFrame(columns=['content', 'topic_id', 'topic_true_name'])
df['content'] = list_docs
df['topic_id'] = targets

def true_topic_name(x, target_names):
    return target_names[x].split('.')[-1]

df['topic_true_name'] = df['topic_id'].apply(lambda x: true_topic_name(x, target_names))
df.head()

# %% [markdown]
# ## Tokenization & preprocessing

# %%
tokenized_data = tokenize(df, form_reduction='stemming', predict=False)

# %%
tokenized_data[['content', 'tokens', 'topic_true_name']].head()

# %%
print("Max number of token:", np.max(tokenized_data.nb_token))
print("Mean number of token:", round(np.mean(tokenized_data.nb_token),2))

# Input format for the model : list of strings (list of tokens)
docs = tokenized_data['tokens'].tolist()
vocab = set(x for doc in docs for x in doc)
n_terms = len(vocab)

print("Voc size:", n_terms)
print("Number of documents:", len(docs))

# %% [markdown]
# ## Training 

# %%
# Train a new model 

# Init of the Gibbs Sampling Dirichlet Mixture Model algorithm
mgp = MovieGroupProcess(K=10, alpha=0.1, beta=0.1, n_iters=30)

vocab = set(x for doc in docs for x in doc)
n_terms = len(vocab)
n_docs = len(docs)

# Fit the model on the data given the chosen seeds
y = mgp.fit(docs, n_terms)

# Save model
with open('dumps/trained_models/model_v2.model', "wb") as f:
    pickle.dump(mgp, f)
    f.close()

# %%
# Load the model used in the post
filehandler = open('dumps/trained_models/model_v1.model', 'rb')
mgp = pickle.load(filehandler)

# %%
doc_count = np.array(mgp.cluster_doc_count)
print('Number of documents per topics :', doc_count)
print('*'*20)

# Topics sorted by document inside
top_index = doc_count.argsort()[-10:][::-1]
print('Most important clusters (by number of docs inside):', top_index)
print('*'*20)


# Show the top 5 words by cluster, it helps to make the topic_dict below
top_words(mgp.cluster_word_distribution, top_index, 5)

# %%
# Must be hand made so the topic names match the above clusters regarding their content
topic_dict = {}
topic_names = ['x',
               'mideast',
               'x',
               'space',
               'space',
               'mideast',
               'space',
               'space',
               'mideast',
               'space']

for i, topic_num in enumerate(top_index):
    topic_dict[topic_num]=topic_names[i]
    
df_pred = topic_attribution(tokenized_data, mgp, topic_dict, threshold=0.4) # threshold can be modified to improve the confidence of the topics



# %%
#pd.set_option('display.max_columns', None)  
#pd.set_option('display.max_colwidth', -1)

df_pred[['content', 'topic_name', 'topic_true_name']].head(20) 

# %%
print("Topic Accuracy:", round(np.sum(np.where((df_pred['topic_true_name'] == df_pred['topic_name']), 1, 0))/len(df_pred), 2))

# %% [markdown]
# # Interactive cluster visualization with pyLDAvis

# %%
# Plot the cluster in a notebook
plot_topic_notebook(tokenized_data, docs, mgp)

# %%
