
# coding: utf-8

# Test Logistic Regression in MLlib

# In[2]:

import pyspark
import numpy as np
from pyspark import SQLContext
from pyspark import SparkContext


# In[3]:

# import dataframe from s3, which will be used as a training dataset
filepath = '/mnt/somefilepath/'
df = sqlContext.read.format("com.databricks.spark.csv")
					.option("header", "true")
					.option("inferschema", "true")
					.option("mode", "DROPMALFORMED")
					.load(filepath)
df.cache()
df.printSchema()


# In[4]:

# set up binomial logistic regression model with elastic net regularization.
# regParam is lambda (>0) and elasticNetParam is alpha (0 <= alpha <= 1)
# alpha and lambda indicates L1 and L2 regularization
# when alpha = 1 --> lasso; when alpha = 0 --> ridge
# regularization prevents overfitting 
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(labelCol="biologic", featuresCol="features", 
						maxIter = 10, regParam = 0.0, elasticNetParam = 0.8)


# In[5]:

# define features and label
from pyspark.ml.feature import VectorAssembler
assembler = (VectorAssembler(inputCols=[x for x in df.columns if x not in ['biologic']],
							outputCol='features'))
df_cleaned = assembler.transform(df)
df_cleaned.cache()


# In[6]:

# fit the model
lrModel = lr.fit(df_cleaned)


# In[7]:

# print coefficients and intercept
print('Coefficient: ', str(lrModel.coefficients))
print('Intercept: ', str(lrModel.intercept))


# In[8]:

df_cleaned.select('features').show(5)


# In[9]:

# Extract the summary from the returned LogisticRegression model instance trained in training df
trainingSummary = lrModel.summary


# In[10]:

# obtain the objective per iteration
objectiveHistory = trainingSummary.objectiveHistory
print('objectiveHistory')
for i in objectiveHistory:
	print(i)


# In[11]:

# obtain the receiver-operating characteristic as a dataframe and areaunderROC
trainingSummary.roc.show()
print('Area Under ROC: ', str(trainingSummary.areaUnderROC))


# In[12]:



