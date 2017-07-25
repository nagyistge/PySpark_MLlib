
# coding: utf-8

# # Model Build Code with Non-MLlib Libraries

# In[2]:

##########################################
# Import libraries used in this notebook #
##########################################


# In[3]:

# import basic pyspark libraries
import pyspark
from pyspark import SQLContext
from pyspark import SparkContext


# In[4]:

# import basic python libraries
import numpy as np
import pandas as pd


# In[5]:

# import model build and evaluation libraries
import statsmodels.api as sm
import sklearn
from sklearn.cross_validation import train_test_split


# In[6]:

# import plotting libraries
import plotly
from plotly.offline import plot
import plotly.graph_objs as go


# In[7]:

######################################################################
# Load dataset from S3 and rename dependent variable column as label #
######################################################################


# In[8]:

# load dataset from S3 and rename dependent variable column 
filepath = 'somefilepathhere'
df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferschema", "true").option("mode", "DROPMALFORMED").load(filepath)
df = df.withColumnRenamed("biologic", "label")
df.cache()
df.printSchema()


# In[9]:

# convert spark dataframe to pandas dataframe
df = df.toPandas()
df.head()


# In[10]:

###################################
# Split training and test dataset #
###################################


# In[11]:

# set test dataframe ratio
test_size = 0.1


# In[12]:

# split train and test dataset
df_train, df_test = train_test_split(df, test_size = test_size)


# In[13]:

########################################################
# Build model: logistic regression with regularization #
########################################################


# In[14]:

# dummify ordinal variables
Presgroup_1_rank = pd.get_dummies(df_train['Presgroup_1'], prefix='Presgroup_1')
Presgroup_2_rank = pd.get_dummies(df_train['Presgroup_2'], prefix='Presgroup_2')
df_train = df_train.join(Presgroup_1_rank.ix[:, 'Presgroup_1':])
df_train = df_train.join(Presgroup_2_rank.ix[:, 'Presgroup_2':])
df_train.head()


# In[15]:

# manually add the intercept
df_train['intercept'] = 1.0
df_train.head()


# In[16]:

# generate features column names and prevent multicolinearity
features = [x for x in list(df_train.columns.values) if x not in ['label','Presgroup_1', 'Presgroup_2','Presgroup_1_0','Presgroup_2_0']]


# In[17]:

alpha = np.ones(len(features), dtype=np.float64)
alpha[-1] = 0 # Don't penalize the intercept
mod = sm.Logit(df_train['label'], df_train[features])
rslt = mod.fit_regularized(alpha=alpha, disp=False)


# In[18]:

print rslt.summary()


# In[19]:

rslt.summary2()


# In[20]:



