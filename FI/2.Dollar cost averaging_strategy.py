# -*- coding: utf-8 -*-
"""
Created on Sun May 24 23:40:45 2020

@author: amatos
from: http://al3xandr3.github.io/Lump-sum-versus-dollar-cost-average-investing.html
"""

## remove this, this is for my personal pc setup
import sys; import os; sys.path.append(os.path.expanduser('~/Google Drive/my/projects/t/'))
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas   as pd
import operator as op
import numpy as np
import seaborn as sns
from datetime import datetime
import random as random
import t as t


# Lets use the S&P500 index as the stock / index we invest in
sp500 = t.get_quotes_close(['^GSPC'], date_from = '2018-02-01', date_to = '2020-02-01')
sp500 = sp500.reset_index()

# Lets have a quick look at the index value over this period
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'png' # svg, png
fig = px.line(sp500, x="date", y="^GSPC", title='S&P 500')
fig.show()

# Lets say we have 1000 to invest
capital = 1000

# Case 1. What happens if we invest everything on the 1st day ?
bulk1 = pd.DataFrame()
bulk1["date"] = sp500["date"]
bulk1["order_size"] = 0
bulk1.loc[t.where(bulk1, "date", datetime.strptime('2018-03-01', '%Y-%m-%d'), op.eq).index[0], "order_size"] = 1000
out1 = t.backtest_strategy(sp500, bulk1, capital)
print(out1["ROI"])
print(out1["IRR"])
print(out1["return"])



# Case 2. What happens if we change investment based on trend


def random_bulk_invest(random_date):
    bulk2 = pd.DataFrame()
    bulk2["date"] = sp500["date"]
    bulk2["order_size"] = 0  
    bulk2.loc[t.where(bulk1, "date", random_date, op.eq).index[0], "order_size"] = 1000
    out2 = t.backtest_strategy(sp500, bulk2, capital)
    return(out2["IRR"], out2["return"])

results = [random_bulk_invest(sp500.sample()["date"].values[0]) for i in range(100)]

_return = [ j for i, j in results ]
t.ci_mean(pd.DataFrame( _return , columns=['bulk_returns'] ), 'bulk_returns')
    

_irr    = [ i for i, j in results if i != -666]
t.ci_mean(pd.DataFrame( _irr , columns=['bulk_returns'] ), 'bulk_returns')


