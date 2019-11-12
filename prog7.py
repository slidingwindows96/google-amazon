#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import csv


# In[2]:


from pgmpy.estimators import MaximumLikelihoodEstimator as mle #important for viva
from pgmpy.models import BayesianModel as bm
from pgmpy.inference import VariableElimination as ve


# In[4]:


heart_data = pd.read_csv("Data7.csv")
heart_data = heart_data.replace("?",np.nan)


# In[5]:


model = bm([("age","trestbps"),
            ("age","fbs"),
            ("sex","trestbps"),
            ("exang","trestbps"),
            ("trestbps","heartdisease"),
            ("fbs","heartdisease"),
            ("heartdisease","restecg"),
            ("heartdisease","thalach"),
            ("heartdisease","chol")])


# In[6]:


model.fit(heart_data,estimator = mle)


# In[7]:


infer = ve(model)
q = infer.query(variables = ["heartdisease"],evidence = {"chol":100})
print(q["heartdisease"])


# In[ ]:




