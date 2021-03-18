# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:02:10 2020

@author: amatos

https://azure.microsoft.com/en-us/blog/root-cause-analysis-with-in-query-machine-learning-in-application-insights-analytics/

"""

import sys; import os; sys.path.append(os.path.expanduser('~/Google Drive/my/projects/t/'))

import operator as op
import numpy as np
import seaborn as sns
from datetime import datetime
import random as random
import t as t
import pandas as pd
from scipy.stats import norm
from scipy import stats

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules




# read test data
df_path = os.path.join('C:\\','Users','amatos','Google Drive','my','projects','ipynb','_data','cluster_es_es.csv')
df = pd.read_csv(df_path, sep=',', error_bad_lines=False)
df = df.drop("EventInfo_Sequence", axis=1)
df = df.drop("Scenario_Status", axis=1)


t2 = pd.DataFrame({'user':['k','j','k','t','k','j'] \
                 , 'period':['before', 'before', 'before', 'before', 'after','after'] \
                 , 'cohort':['control','control','control','control','control','control'] \
        , 'isLabel':['failure','failure','success','success','success','success']         
        })

# ---------------------------------------


## Input Interface
Dataset = t2
Column  = "isLabel"
ClassificationA = "failure"
ClassificationB = "success"
# Other filtering parameters


cola = f"{Column}:{ClassificationA}"
colb = f"{Column}:{ClassificationB}"

# ---------------------------------------

# dataset cleanup
Dataset = Dataset.fillna("NA")


## Transform into columns
#dt = df.values.tolist()
#te     = TransactionEncoder()
#te_ary = te.fit(dt).transform(dt)
#df     = pd.DataFrame(te_ary, columns=te.columns_)


df = pd.get_dummies(Dataset, prefix_sep=":")


# frequent item sets, in same basket (with a min support)
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.01)

cola_rules = t.where(rules, "consequents", {cola})

# Count occurences
for index, row in cola_rules.iterrows():
    #print(list(row['antecedents']), list(row['consequents']))
    valu = t.where(df, cola, True)
    #print(valu.shape[0])
    for ant in list(row['antecedents']):
        valu  = t.where(valu, ant, True)
    cola_rules.at[index,'A_cnt'] = valu.shape[0]
    
    valu2 = t.where(df, colb, True)
    for ant in list(row['antecedents']):
        valu2 = t.where(valu2, ant, True)
    cola_rules.at[index,'B_cnt'] = valu2.shape[0]
    
cola_rules["A_ratio"] = cola_rules["A_cnt"] / (cola_rules["B_cnt"]+cola_rules["A_cnt"])


output = t.sort(cola_rules, "A_ratio", ascending=False)

output = t.relabel(output, "A_cnt", f"{ClassificationA}_cnt")
output = t.relabel(output, "B_cnt", f"{ClassificationB}_cnt")
output = t.relabel(output, "A_ratio", f"{ClassificationA}_ratio")

#success  = t.where(rules, "consequents", {"success"})


# TODO
# - make a function, 
# - add it into library
# - make a post on it
