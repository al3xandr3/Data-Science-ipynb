# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Large Data with SQLite

# %% [markdown]
# ## Import Libraries

# %%
## https://github.com/al3xandr3/T
import sys; import os; sys.path.append(os.path.expanduser('~/DropBox/my/projects/T/'))

# %%
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import pandas   as pd
import operator as op

import numpy             as np
import seaborn           as sns
from datetime import datetime

# References
# https://pythonspeed.com/articles/indexing-pandas-sqlite/
# http://al3xandr3.github.io/table-query.html

# %% [markdown]
# ## Class definition

# %%
import pandas as pd
import sqlite3

class db:
    
    db_path    = ""
    table_name = ""
    csv_path   = ""
    _db = ""
    
    def __init__(self, db_path="C:/Users/amatos/data/database.sqlite", table_name="tabl"):

        self.db_path = db_path
        self.table_name = table_name
        try:
            self._db = sqlite3.connect(db_path)
            print("sqlite3.version: " + sqlite3.version)
        except Error as e:
            print("Could not Connect to Database\n")
            print(e)

    def __del__(self):
        self._db.close()

    def execute(self, query):
        cur = self._db.cursor()
        cur.execute(query)

    def query(self, query):
        return pd.read_sql_query(query, self._db)

    def tables(self):
        return(self.query("select name from sqlite_master where type='table' ORDER BY name;"))

    def import_csv (self, csvfile_path, table_name="", index="", **kwargs):
        database_path = self.db_path 
        db = self._db
        table_name= table_name or self.table_name
        self.execute(f"drop table if exists {table_name};")

        # Load the CSV in chunks:
        for c in pd.read_csv(csvfile_path, chunksize=1000):
            # Append all rows to a new database table
            c.to_sql(table_name, db, if_exists="append")
        # Add an index on the 'street' column:
        #db.execute("CREATE INDEX street ON <table_name>(<index>)") 


# %% [markdown]
# ## Testing / Using

# %%
d = db()

path = "https://raw.githubusercontent.com/curran/data/gh-pages/oecd/houseprices-oecd-1975-2012.csv"
d.import_csv(path)

d.tables()

d.execute("drop table if exists tabl;")

d.query(r"SELECT * FROM tabl LIMIT 5;")
