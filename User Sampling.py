# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:02:10 2020

@author: amatos
"""

import sys; import os; sys.path.append(os.path.expanduser('~/Google Drive/my/projects/t/'))
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import operator as op
import numpy as np
import seaborn as sns
from datetime import datetime
import random as random
import t as t
import pandas as pd
from scipy.stats import norm
from scipy import stats
import uuid
from scipy.stats import chisquare

"""
# Goal

Lets say we need to sample 10% users, with properties:
- Always put same user into same bucket (either sampled or unsampled)
- Fast, we don't have to memorize (keep track) of users already sampled
- Assures a 10% even distribution of users 

## Hypothesis
    If guid numbers are randomly distributed we can use a mod operation to select 10% of users    
 
## Planning

- This notebook uses randomly generated GUIDs to test if this method has the properties that we require (it does)
- Next step is to get a collection of real user GUIDs from the system, apply this function and sense check if they show an even distribution
- Furthermore, a 3rd step could be apply this method 1 days of users (sampling 10% of them) calculate the KPIs and see if numbers make sense, also as a simulation of how they look like when in production.
"""


"""
## GUIDs as user ID

GUID, version 4, looks like this:
xxxxxxxx-xxxx-4xxx-xxxx-xxxxxxxxxxxx

- It has 31 chars that are random (i.e. provides 122 bits of randomness)
- Randomness masks the process of generating the GUID and thus helps making the sampling fair
- We can test for even distribution once we apply the real user id's that we have in the system
"""

# uuids = MS guids
print(uuid.uuid4())

# generate a collection of them and convert to int
users = pd.DataFrame()
users["user_id"] = [uuid.uuid4().int for i in range(1000)]
users.head()


"""
## Test they are distributed evenly

- Create a 10 buckets distribution
- This sampling depends on the fact that guids are generated at source randomly (i.e. with equal distribution)
"""

def is_sample_bucket(user_id):
    return (user_id % 10 == 0) # we can pick any number between 0 and 9

users["is_sampled"] = is_sample_bucket(users["user_id"].values)
users.head()

sg = t.group(users, "is_sampled")
sg["%"] = sg["is_sampled_count"] / np.sum(sg["is_sampled_count"]) * 100
sg

# generate buckets for further validation is doing the right thing
# lets us check if generation at source random enought for an even distribution
def sample_bucket(user_id):
    return (user_id % 10)

def bucket_stats(values):
    nr = pd.DataFrame(values, columns=["IDs"])
    nr["sample_bucket"] = sample_bucket(nr["IDs"].values)
    nrs = t.group(nr, "sample_bucket")
    nrs = t.relabel(nrs.reset_index(), "index", "bucket")
    nrs["%"] = nrs["sample_bucket_count"] / np.sum(nrs["sample_bucket_count"]) * 100
    nrs = t.sort(nrs, "bucket")
    return nrs

bd = bucket_stats(users["user_id"].values)

"""
## Consistent bucket?
    
- Another property required, is that we need to apply the function above and keep putting the same user into same bucket.
- So we can assure we continue capture events from the same user over time
- This needs to happen without having to save the already sampled users into a database and then retrieve them when we are deciding who to include them in sampling. This operation would be too expensive.
  

So here we check that user always goes into same bucket
i.e. run again same function and see if previously selected user goes into same bucket
"""

results = []
for i in range(0, 10000):
    random_user = users.loc[random.randint(0, len(users.index)-1), "user_id"]
    random_user_previous_bucket = t.select(t.where(users, "user_id", random_user), "sample_bucket").values[0][0]
    results.append(random_user_previous_bucket == sample_bucket(random_user))

r = pd.DataFrame(results, columns=["result"])
t.group(r, "result")


"""
# Testing that GUIDs numbers have a truly  random distribution

- We can see this already when we do the counts above by bucket, where we expect 100 into each bucket
- But another way, is to run a Shapiro–Wilk test is a test of normality.
- The null hypothesis is: the set of numbers originate from a normal distribution (not a random one)

results: 
- if the p-value<0.05 then is a random distribution
- if the p-value>0.05 then is not random

Also we can see what non-random looks like
"""

# Non-Random numbers

# Sequence
seq = range(1000)
bucket_stats(sequence)
_, p_value_nr = stats.shapiro(sequence)
print("p-value for non-random sample: " + str(p_value_nr))


# Binomial
bino = np.random.binomial(n=100, p=0.5, size=1000)
bucket_stats(bino)
plt.hist(bino, bins=20, normed=True)
plt.show()
_, p_value_bi = stats.shapiro(bino)
print("p-value for non-random sample: " + str(p_value_bi))
chisquare(bino)[1]

# test on something that we know is not random:
not_random_data = norm.rvs(size=1000) # sample from a normal distribution
ks_statistic, p_value_nr = stats.shapiro(not_random_data)
print("p-value for non-random sample: " + str(p_value_nr))


# p-value of non random for comparison
not_random_data = norm.rvs(size=1000)
ks_statistic, p_value_nr = stats.shapiro(not_random_data)
print("p-value for non-random sample: " + str(p_value_nr))
