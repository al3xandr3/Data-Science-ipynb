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
print(str(ks_statistic) + " P-value for non-random sample: " + str(p_value_nr))


random_data = np.random.randint(1000, size=1000)
shapiro_statistic, p_value_r = stats.shapiro(random_data)
print(str(ks_statistic) + " P-value for random sample: " + str(p_value_r))


list_random_uuids = [uuid.uuid4().int for i in range(1000) ]
shapiro_statistic, p_value_rg = stats.shapiro(list_random_uuids)
print("P-value for random sample of guids: " + str(p_value_rg))


b = [389752 ,387360 ,388595 ,388442 ,389793 ,387986 ,387921 ,387868 ,388581 ,387830]
a = [1, 387360 , 50,3884420 ,3897 ,0 ,38792 ,386 ,3885810 , 3287999830, 1, 2, 34, 1]


values = b
ks_statistic, p_value_nr = stats.kstest(values, 'norm', args=(np.mean(values), np.std(values)))
print("P-value: " + str(p_value_nr) + " Distribution is: "+ ("Random" if p_value_nr<=0.05 else "non-Random"))



