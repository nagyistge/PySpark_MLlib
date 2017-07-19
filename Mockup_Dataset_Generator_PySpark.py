
# coding: utf-8

# Mockup Dataset Generator on pysparks

# In[2]:

import pyspark
import numpy as np
import pandas as pd
from pyspark import SQLContext
from pyspark.mllib.random import RandomRDDs
from pyspark.sql.functions import udf


# In[3]:
# define sample dataset size n
n = 10000

# test the speed of RDD random generator
gender_1 = RandomRDDs.normalRDD(sc, n, seed = 1).map(lambda x: np.round(x))
gender_1.count()


# In[4]:

# test the speed of numpy random generator
gender_2 = np.random.randint(0,2,size = n)
len(gender_2)


# For the size n, numpy is faster than spark RDD. Use numpy to build arrays 
# then convert to pyspark dataframe.

# In[6]:

# create numpy arrays 
biologic = np.random.randint(0,2,size = n)
gender = np.random.randint(0,2,size = n)
age = np.random.randint(18,85,size = n)
num_ocs_1 = np.random.randint(0,12,size = n)
days_last_ocs = np.random.randint(0,365,size = n)
num_ics_1 = np.random.randint(0,12,size = n)
days_last_ics = np.random.randint(0,365,size = n)
num_laba_1 = np.random.randint(0,12,size = n)
days_last_laba = np.random.randint(0,365,size = n)
pres_1 = np.random.randint(0,2,size = n)
pres_2 = np.random.randint(0,2,size = n)
presgroup_1 = np.random.randint(0,4,size = n)
presgroup_2 = np.random.randint(0,4,size = n)
payer_1 = np.random.randint(0,2,size = n)
payer_2 = np.random.randint(0,2,size = n)
alle_1 = np.random.randint(0,2,size = n)
alle_2 = np.random.randint(0,2,size = n)
eos_max_1 = np.random.randint(0,36,size = n)
days_eos = np.random.randint(0,365,size = n)
cop_d = np.random.randint(0,2,size = n)


# In[7]:

col_names = ['biologic', 'gender', 'age', 'num_ocs_1', 'time_ocs',  
			'num_ICS_1', 'time_ics', 'num_laba_1', 'time_laba', 'Pres_1', 'Pres_2', 
			'Presgroup_1', 'Presgroup_2',  'Payer_1',  'Payer_2', 'Alle_1', 'Alle_2', 
			'EoS_1', 'Time_eos_1', 'COP_D']

p_df = pd.DataFrame({col_names[0]:biologic,col_names[1]:gender,col_names[2]:age,
					col_names[3]:num_ocs_1,col_names[4]:days_last_ocs,col_names[5]: num_ics_1,
					col_names[6]:days_last_ics,col_names[7]:num_laba_1,
					col_names[8]:days_last_laba,col_names[9]: pres_1,col_names[10]:pres_2,
					col_names[11]:presgroup_1,col_names[12]:presgroup_2,col_names[13]:payer_1,
					col_names[14]:payer_2,col_names[15]:alle_1,col_names[16]:alle_2,
					col_names[17]:eos_max_1,col_names[18]:days_eos,col_names[19]:cop_d})
p_df.head()


# In[8]:

sqlCtx = SQLContext(sc)
df = sqlCtx.createDataFrame(p_df)
df.show()


# In[9]:

# save dataframe to s3 as csv files (with partition)
filepath = '/mnt/somefilepath'
df.write.format('com.databricks.spark.csv').option('header','True').save(filepath)


# In[10]:



