# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:02:10 2020

@author: amatos
"""

from scipy.stats import norm
from scipy import stats
import numpy  as np
import uuid


# Null Hypothesys here is that a set of numbers is normally 
# distributed (i.e. non-random)

not_random_data = norm.rvs(size=1000)
ks_statistic, p_value_nr = stats.shapiro(not_random_data)
print("P-value for non-random sample: " + str(p_value_nr))


random_data = np.random.randint(1000, size=1000)
shapiro_statistic, p_value_r = stats.shapiro(random_data)
print("P-value for random sample: " + str(p_value_r))


list_random_uuids = [uuid.uuid4().int for i in range(1000) ]
shapiro_statistic, p_value_rg = stats.shapiro(list_random_uuids)
print("P-value for random sample of guids: " + str(p_value_rg))

