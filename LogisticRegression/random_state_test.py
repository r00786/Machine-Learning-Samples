# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:17:32 2019

@author: Rohit
"""

from sklearn.model_selection import train_test_split
import numpy as np

a, b = np.arange(10).reshape((5, 2)), range(5)


print(train_test_split(a,b,random_state=32))
