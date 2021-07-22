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
# # K-prototype Cluster Algorithm
# https://towardsdatascience.com/the-k-prototype-as-clustering-algorithm-for-mixed-data-type-categorical-and-numerical-fe7c50538ebb
# %% [markdown]
# ---

# %% [markdown]
# ## Import module

# %%
# Import module for data manipulation
import pandas as pd
# Import module for linear algebra
import numpy as np
# Import module for data visualization
from plotnine import *
import plotnine

# Import module for k-protoype cluster
from kmodes.kprototypes import KPrototypes

# %%
# Ignore warnings
import warnings
warnings.filterwarnings('ignore', category = FutureWarning)

# %%
# Format scientific notation from Pandas
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# %% [markdown]
# ## Import data

# %% [markdown]
# Data source: http://eforexcel.com/wp/downloads-18-sample-csv-files-data-sets-for-testing-sales/

# %%
df = pd.read_csv('data/10000 Sales Records.csv')

# %%
print('Dimension data: {} rows and {} columns'.format(len(df), len(df.columns)))
df.head()

# %%
df.columns

# %%
df.info()

# %%
df.select_dtypes('object').nunique()

# %%
# Summary statistics of numerical variable
for i in df.select_dtypes('object').columns:
    print(df[i].value_counts(),'\n')

# %%
df.describe()

# %%
# Check missing value
df.isna().sum()

# %% [markdown]
# ## Explanatory data analysis

# %% [markdown]
# #### Distribution of region

# %%
df_region = pd.DataFrame(df['Region'].value_counts()).reset_index()
df_region['Percentage'] = df_region['Region'] / df['Region'].value_counts().sum()
df_region.rename(columns = {'index':'Region', 'Region':'Total'}, inplace = True)
df_region = df_region.sort_values('Total', ascending = True).reset_index(drop = True)
df_region

# %%
df_region = df.groupby('Region').agg({
    'Region': 'count',
    'Units Sold': 'mean',
    'Total Revenue': 'mean',
    'Total Cost': 'mean',
    'Total Profit': 'mean'
    }
).rename(columns = {'Region': 'Total'}).reset_index().sort_values('Total', ascending = True)

# %%
df_region

# %%
plotnine.options.figure_size = (8, 4.8)
(
    ggplot(data = df_region)+
    geom_bar(aes(x = 'Region',
                 y = 'Total'),
             fill = np.where(df_region['Region'] == 'Asia', '#981220', '#80797c'),
             stat = 'identity')+
    geom_text(aes(x = 'Region',
                   y = 'Total',
                   label = 'Total'),
               size = 10,
               nudge_y = 120)+
    labs(title = 'Region that has the highest purchases')+
    xlab('Region')+
    ylab('Frequency')+
    scale_x_discrete(limits = df_region['Region'].tolist())+
    theme_minimal()+
    coord_flip()
)

# %% [markdown]
# #### Distribution of item type

# %%
# Order the index of cross tabulation
order_region = df_region['Region'].to_list()
order_region.append('All')
order_region

# %%
df_item = pd.crosstab(df['Region'], df['Item Type'], margins = True).reindex(order_region, axis = 0).reset_index()
# Remove index name
df_item.columns.name = None
df_item

# %% [markdown]
# ## Data pre-processing

# %% [markdown]
# #### Remove unused columns for the next analysis `Country`, `Order Date`, `Order ID`, and `Ship Date`

# %%
df.drop(['Country', 'Order Date', 'Order ID', 'Ship Date'], axis = 1, inplace = True)

# %%
print('Dimension data: {} rows and {} columns'.format(len(df), len(df.columns)))
df.head()

# %% [markdown]
# ## Cluster analysis

# %% [markdown]
# Thanks to https://github.com/aryancodify/Clustering

# %%
# Get the position of categorical columns
catColumnsPos = [df.columns.get_loc(col) for col in list(df.select_dtypes('object').columns)]
print('Categorical columns           : {}'.format(list(df.select_dtypes('object').columns)))
print('Categorical columns position  : {}'.format(catColumnsPos))

# %%
dfMatrix = df.to_numpy()

# %%
dfMatrix

# %% [markdown]
# Error of initialization: https://github.com/nicodv/kmodes/blob/master/README.rst#faq

# %%
# Choosing optimal K
cost = []
for cluster in range(1, 10):
    try:
        kprototype = KPrototypes(n_jobs = -1, n_clusters = cluster, init = 'Huang', random_state = 0)
        kprototype.fit_predict(dfMatrix, categorical = catColumnsPos)
        cost.append(kprototype.cost_)
        print('Cluster initiation: {}'.format(cluster))
    except:
        break

# %%
# Converting the results into a dataframe and plotting them
df_cost = pd.DataFrame({'Cluster':range(1, 6), 'Cost':cost})
df_cost.head()

# %%
plotnine.options.figure_size = (8, 4.8)
(
    ggplot(data = df_cost)+
    geom_line(aes(x = 'Cluster',
                  y = 'Cost'))+
    geom_point(aes(x = 'Cluster',
                   y = 'Cost'))+
    geom_label(aes(x = 'Cluster',
                   y = 'Cost',
                   label = 'Cluster'),
               size = 10,
               nudge_y = 1000) +
    labs(title = 'Optimal number of cluster with Elbow Method')+
    xlab('Number of Clusters k')+
    ylab('Cost')+
    theme_minimal()
)

# %%
# Fit the cluster
kprototype = KPrototypes(n_jobs = -1, n_clusters = 3, init = 'Huang', random_state = 0)
kprototype.fit_predict(dfMatrix, categorical = catColumnsPos)

# %%
# Cluster centorid
kprototype.cluster_centroids_

# %%
# Check the iteration of the clusters created
kprototype.n_iter_

# %%
# Check the cost of the clusters created
kprototype.cost_

# %%
# Add the cluster to the dataframe
df['cluster_id'] = kprototype.labels_ 

# %%
df_region = pd.DataFrame(df['Region'].value_counts()).reset_index()
df_region['Percentage'] = df_region['Region'] / df['Region'].value_counts().sum()
df_region.rename(columns = {'index':'Region', 'Region':'Total'}, inplace = True)
df_region = df_region.sort_values('Total', ascending = True).reset_index(drop = True)
df_region
