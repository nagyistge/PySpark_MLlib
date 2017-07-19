
# coding: utf-8

# Cross Validation Testing

# In[2]:

import pyspark
import numpy as np
from pyspark import SQLContext
from pyspark import SparkContext
from pyspark.ml import Pipeline 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


# In[3]:

# import dataframe from s3
filepath = 'somefilepathhere'
df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferschema", "true").option("mode", "DROPMALFORMED").load(filepath)
df = df.withColumnRenamed("biologic", "label")
df.cache()
df.printSchema()


# In[4]:

# split df into training and test dataset
df_train, df_test = df.randomSplit([0.9,0.1], seed = 123)


# In[5]:

# set up binomial logistic regression model with elastic net regularization.
# regParam is lambda (>0) and elasticNetParam is alpha (0 <= alpha <= 1)
# alpha and lambda indicates L1 and L2 regularization
# when alpha = 1 --> lasso; when alpha = 0 --> ridge
# regularization prevents overfitting 
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(labelCol="label", featuresCol="features")


# In[6]:

# define features and label
from pyspark.ml.feature import VectorAssembler
assembler = (VectorAssembler(inputCols=[x for x in df_train.columns if x not in ['biologic']],outputCol='features'))


# In[7]:

# use pipeline to configure ML
pipeline = Pipeline(stages = [assembler, lr])
paramGrid = ParamGridBuilder().addGrid(lr.maxIter, [10,15]).addGrid(lr.regParam, [0.0, 0.1]).addGrid(lr.elasticNetParam, [0.9, 0.8]).build()


# In[8]:

# use pipeline as the extimator in cross validation
# add function to choose different evaluator
crossval = CrossValidator(estimator = pipeline, estimatorParamMaps = paramGrid, evaluator = BinaryClassificationEvaluator(), numFolds = 3)
cvModel = crossval.fit(df_train)


# In[9]:

import pandas as pd
# find the best model
best = cvModel.bestModel.stages[1]
bestSum = best.summary
roc = bestSum.roc.toPandas()
roc.head()


# In[10]:

# use plotly to generate graph for ROC curve
import plotly
from plotly.offline import plot
import plotly.graph_objs as go
lw = 2

trace1 = go.Scatter(x=roc['FPR'], y=roc['TPR'], mode='lines', line=dict(color='darkorange', width=lw),name='ROC curve (area = %0.2f)' %bestSum.areaUnderROC)

trace2 = go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='navy', width=lw, dash='dash'),showlegend=False)

layout = go.Layout(title='Receiver operating characteristic example',xaxis=dict(title='False Positive Rate'),yaxis=dict(title='True Positive Rate'))

fig = plot(go.Figure(data=[trace1, trace2], layout=layout), output_type='div')


# In[11]:

# display ROC curve
displayHTML(fig)


# In[12]:




# In[13]:




# In[14]:



