# -*- coding: utf-8 -*-
"""
Created on Sun May 24 23:40:45 2020

@author: amatos
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


"""
Lump vs dollar-cost average investing


As the market goes up and down the moment you invest can be crucial, 
you don't want invest all your money at the wrong time  
so and alternative strategy is spread out the investment over time, 
so that sometimes you invest in high's sometimes int he lows 
but overtime it averages out. And has the side effect of not having to worry and time it right
On the other side the earlier you invest the more time you are allowing money to
grow as an investment (the compounding effect).

So what is best ? to invest all now or invest instead in smaller quantities overtime ?

"""

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

# What happens if we invest everything on the 1st day ?
bulk1 = pd.DataFrame()
bulk1["date"] = sp500["date"]
bulk1["order_size"] = 0
bulk1.loc[t.where(bulk1, "date", datetime.strptime('2018-03-01', '%Y-%m-%d'), op.eq).index[0], "order_size"] = 1000
out1 = t.backtest_strategy(sp500, bulk1, capital)
print(out1["lift"])
print((out1["lift"]*1000)+1000)

# 0.20459956366564758
# 1204.5995636656476

# What happens if we pick a random date to invest everything ?
def random_bulk_invest(random_date):
    bulk2 = pd.DataFrame()
    bulk2["date"] = sp500["date"]
    bulk2["order_size"] = 0  
    bulk2.loc[t.where(bulk1, "date", random_date, op.eq).index[0], "order_size"] = 1000
    out2 = t.backtest_strategy(sp500, bulk2, capital)
    return((out2["lift"]*1000)+1000)

results = [random_bulk_invest(sp500.sample()["date"].values[0]) for i in range(100)]

#fig = px.histogram(x=results, nbins=20)
#fig.show()

df = pd.DataFrame( results , columns=['bulk_returns'] )
t.ci_mean(df, 'bulk_returns')
    

# Random date within the first 60 days
results2 = [random_bulk_invest(sp500.head(60).sample()["date"].values[0]) for i in range(100)]
df = pd.DataFrame( results2 , columns=['bulk_returns'] )
t.ci_mean(df, 'bulk_returns')
    

##########
# Now lets try the Dollar-Cost Averaging investing approach
# here we invest smaller vallues at a time

# For example a 50 every 7 days, until money runs out, starting earlier 
dca = pd.DataFrame()
dca["date"] = sp500["date"]
dca["order_size"] = ([50,0,0,0,0,0,0]*20) + [0]*(sp500["date"].size-140)
out3 = t.backtest_strategy(sp500, dca, capital)
print(round(out3["lift"]*100,1), round(out3["lift"]*1000, 1)+1000)
# > 17.3 1173.1


# For e.g. invest 50 every 7 days, until money runs out, but start later 
dca = pd.DataFrame()
dca["date"] = sp500["date"]
dca["order_size"] = ( [0]*(sp500["date"].size-140) + [50,0,0,0,0,0,0]*20) 
np.sum(dca["order_size"])
out3 = t.backtest_strategy(sp500, dca, capital)
print(round(out3["lift"]*100,1), round(out3["lift"]*1000, 1)+1000)
# > 5.7 1056.6


# Or a 200 every 28 days, until money runs out
dca = pd.DataFrame()
dca["date"] = sp500["date"]
dca["order_size"] = (([100]+[0]*27)*10) + [0]*(sp500["date"].size-280)
np.sum(dca["order_size"])
out3 = t.backtest_strategy(sp500, dca, capital)
print(round(out3["lift"]*100,1), round(out3["lift"]*1000, 1)+1000)
# > 17.4 1174.4



# or randomly inserting a random quantity
def random_dca_invest(early=False):
    dca = pd.DataFrame()
    dca["date"] = sp500["date"]
    dca["order_size"] = 0
    money = capital
    while money > 0:
        # up to 50 each time
        random_invest_value = random.randint(1, 50) if money > 50 else money
        random_day =""
        if early:
            random_day = t.where(dca, "order_size", 0).head(60).sample()["date"].values[0]
        else:
            random_day = t.where(dca, "order_size", 0).sample()["date"].values[0]
        dca.loc[t.where(dca, "date", random_day, op.eq).index[0], "order_size"] = random_invest_value    
        money = money - random_invest_value

    out3 = t.backtest_strategy(sp500, dca, capital)
    return((out3["lift"]*1000)+1000)


results = [random_dca_invest(early=True) for i in range(100)]

df = pd.DataFrame( results , columns=['bulk_returns'] )
t.ci_mean(df, 'bulk_returns')

## Anytime 
# {'mean': 1132.124380136475,
# '95% conf int of mean': array([1129.92337399, 1134.32579255])}

# Early
#{'mean': 1197.9713622371173,
# '95% conf int of mean': array([1197.2692635 , 1198.69807711])}

#fig = px.histogram(x=results, nbins=20)
#fig.show()

# Futher Reading: 
# - https://www.investopedia.com/articles/stocks/07/dcavsva.asp
# - https://awealthofcommonsense.com/2018/05/the-lump-sum-vs-dollar-cost-averaging-decision/#:~:text=On%20average%2C%20the%20lump%20sum,percent%2C%20depending%20on%20the%20country.&text=I%20compared%20a%20lump%20sum,was%20at%2032%20or%20higher.



