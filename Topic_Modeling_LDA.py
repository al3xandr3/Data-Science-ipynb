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

# %% [markdown]
# # Import libraries

# ## References
# LDA explained https://youtu.be/Cpt97BpI-t4
# https://www.datacamp.com/community/tutorials/stemming-lemmatization-python
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

import os
# %%
## remove this, this is for my personal pc setup
import sys

sys.path.append(os.path.expanduser("~/Google Drive/my/projects/t/"))

# %%
import matplotlib
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

import operator as op
import re
import string
from datetime import datetime

# import the nltk package
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
# %%
import spacy
import t as t  # want T to be accessible
from IPython.display import IFrame
# call the nltk downloader
# nltk.download()
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
# %%
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# %load_ext autoreload
# %autoreload

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'


# %% [markdown]
# # Read in the data
#
# News groups headlines
#
# Reference; https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/


def twenty_newsgroup():
    newsgroups_train = fetch_20newsgroups(
        subset="train", remove=("headers", "footers", "quotes")
    )

    df = pd.DataFrame([newsgroups_train.data, newsgroups_train.target.tolist()]).T
    df.columns = ["text", "target"]

    targets = pd.DataFrame(newsgroups_train.target_names)
    targets.columns = ["title"]

    out = pd.merge(df, targets, left_on="target", right_index=True)
    out["date"] = pd.to_datetime("now")
    return out


df = twenty_newsgroup()

# %%
# from io import StringIO

# %%
# df = pd.read_csv(StringIO(sampled_data), encoding='utf8', sep=",", parse_dates=True)

# %%
df.head()

# %%
# ETL Sample

# %%
import datetime


def to_date(row, column):
    try:
        print(row[column].split(".")[0])
        return datetime.datetime.strptime(row[column].split(".")[0], "%Y-%m-%d %H:%M:%S")
    except:
        return datetime.datetime.strptime("0:0:1", "%H:%M:%S")


# df['date2'] = df.apply(lambda x : to_date(x, "date"),axis =1)

# %%
# drop unused column
df.drop(["date", "title", "target"], axis=1, inplace=True)

# %%
# remove empties
print("(Rows, Columns)\n", df.shape)
df = df.dropna(subset=["text"])
print("(Rows, Columns)\n", df.shape)

# %%
# fill na's
df["text"].fillna("", inplace=True)


# %%
def clean_text(text):
    """Make text lowercase, remove punctuation and remove words containing numbers."""
    # camel = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', text)
    # text = " ".join(camel)
    text = text.lower()
    # text = re.sub(r'\[.*?\]', '', text) # remove text in square brackets
    text = re.sub(r"[\W_]+", " ", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d*", "", text)  # remove numbers
    text = re.sub(r"\s+", " ", text)  # remove multiple spaces
    text = text.strip()
    return text


df["text_clean"] = df.text.apply(lambda x: clean_text(x))

# %%
df.head()

# %% https://www.datacamp.com/community/tutorials/stemming-lemmatization-python
column_target = "text_clean"
df[column_target].fillna("", inplace=True)
porter = PorterStemmer()


def stemmetizer(text):
    sent = []
    doc = word_tokenize(text)
    for word in doc:
        sent.append(porter.stem(word))
    return " ".join(sent)


df["stem"] = df.apply(lambda x: stemmetizer(x[column_target]), axis=1)

# %%

text_col = "stem"

# %%
# EDA

# %%
import plotly.express as px

# Distribution of the count of words we are using
plt.figure(figsize=(10, 6))
doc_lens = [len(d) for d in df[text_col]]
counts, bins = np.histogram(doc_lens, bins=range(0, 50, 1))
bins = 0.5 * (bins[:-1] + bins[1:])
fig = px.bar(x=bins, y=counts, labels={"x": "words", "y": "count"})
fig.write_html("plot.html", auto_open=True)

# Wordcould of words we are using
from subprocess import check_output

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS, WordCloud

mpl.rcParams["figure.figsize"] = (12.0, 12.0)
mpl.rcParams["font.size"] = 12
mpl.rcParams["savefig.dpi"] = 100
mpl.rcParams["figure.subplot.bottom"] = 0.1
stopwords = set(STOPWORDS)

wordcloud = WordCloud(
    background_color="white",
    stopwords=stopwords,
    max_words=50,
    max_font_size=40,
    random_state=42,
).generate(" ".join(df[text_col].values))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# %%
import plotly.graph_objects as go
from plotly.offline import plot


# Count of unigrams
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words="english").fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


common_words = get_top_n_words(df[text_col], 30)
df2 = pd.DataFrame(common_words, columns=["unigram", "count"])

fig = go.Figure([go.Bar(x=df2["unigram"], y=df2["count"])])
fig.update_layout(title=go.layout.Title(text="Count of unigrams"))
fig.write_html("plot.html", auto_open=True)

# %%

# Count of Bigrams
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words="english").fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


common_words = get_top_n_bigram(df[text_col], 20)
df3 = pd.DataFrame(common_words, columns=["bigram", "count"])

fig = go.Figure([go.Bar(x=df3["bigram"], y=df3["count"])])
fig.update_layout(title=go.layout.Title(text="Count of Bigrams"))
fig.write_html("plot.html", auto_open=True)

# %%


# Count of trigrams
def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words="english").fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


common_words = get_top_n_trigram(df[text_col], 20)
df4 = pd.DataFrame(common_words, columns=["trigram", "count"])

fig = go.Figure([go.Bar(x=df4["trigram"], y=df4["count"])])
fig.update_layout(title=go.layout.Title(text="Count of Trigrams"))
fig.write_html("plot.html", auto_open=True)


# %%
# Prep data

vectorizer = CountVectorizer(
    analyzer="word",
    max_df=0.95,
    min_df=3,  # words occurring in only one document or in at least 95% of the documents are removed.
    stop_words="english",
    lowercase=True,
    token_pattern="[a-zA-Z0-9]{3,}",
    max_features=5000,
)

data_vectorized = vectorizer.fit_transform(df[text_col])

# %%
# LDA Grid Search 
"""
from sklearn.model_selection import GridSearchCV

# Define Search Param
search_params = {"n_components": [2, 5, 10, 15, 20], "learning_decay": [0.5, 0.7, 0.9]}

# Init the Model
lda = LatentDirichletAllocation(
    max_iter=5, learning_method="online", learning_offset=50.0, random_state=0
)

# Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)
# Do the Grid Search
model.fit(data_vectorized)

model.best_params_
"""

# %%
# LDA

lda_model = LatentDirichletAllocation(
    n_components=10,  # Number of topics
    learning_method="online",
    random_state=0,
    learning_decay=0.9,
    n_jobs=-1,  # Use all available CPUs
)

lda_output = lda_model.fit_transform(data_vectorized)


# %%
# LDA gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel# spaCy for preprocessing

# Create Dictionary
id2word = corpora.Dictionary(df[text_col])

# Create Corpus
#texts = df[text_col]

# Term Document Frequency
#corpus = [id2word.doc2bow(text) for text in texts]

# View
#print(corpus[:1])

#lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                           id2word=id2word,
#                                           num_topics=20, 
#                                           random_state=100,
#                                           update_every=1,
#                                           chunksize=100,
#                                           passes=10,
 #                                          alpha='auto',
#                                           per_word_topics=True)
#
#lda_output = lda_model.fit_transform(data_vectorized)

# %% Viz
import pyLDAvis
import pyLDAvis.sklearn

# pyLDAvis.enable_notebook()
p = pyLDAvis.sklearn.prepare(lda_model, data_vectorized, vectorizer, mds="tsne")
pyLDAvis.save_html(p, "c:/tmp/lda.html")


#%% Top keywords for each topic

# Show top n keywords for each topic
def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords


topic_keywords = show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=15)

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ["Word " + str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ["Topic " + str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords


# %%

# %%
# Topic-Keyword Matrix
topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]
df_topic_keywords = pd.DataFrame(lda_model.components_)

# Assign Column and Index
df_topic_keywords.columns = vectorizer.get_feature_names()
df_topic_keywords.index = topicnames

# View
df_topic_keywords.head()

# %%
lda_model.perplexity(data_vectorized)
# Note that the gensim lib has better metrics, we could use the gensim even to find ideal number of clusters using coherence metric, and then switch to sklearn for rest.

# %% [markdown]
# For shorter text modeling, assuming only 1 topic per text see:
# https://towardsdatascience.com/short-text-topic-modelling-lda-vs-gsdmm-20f1db742e14
# https://towardsdatascience.com/short-text-topic-modeling-70e50a57c883
