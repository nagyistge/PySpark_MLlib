
# coding: utf-8

# **Logistic Regression model build with cross validation and evaluation**

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
from pyspark.ml import Pipeline 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics


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

###################################
# Split training and test dataset #
###################################


# In[10]:

# set training, testing ratio and split seed number
train_ratio = 0.9
test_ratio = 1 - train_ratio
split_seed = 123


# In[11]:

# split dataframe
df_train, df_test = df.randomSplit([train_ratio,test_ratio], seed = split_seed)


# In[12]:

##########################################################
# Build model: logistic regression with cross validation #
##########################################################


# In[13]:

#################################################################################
# Set up feature transformation and logistic regression in pipeline (estimator) #
#################################################################################


# In[14]:

# set up the assembler to create the features vector for logistic regression
assembler = (VectorAssembler(inputCols=[x for x in df_train.columns if x not in ['label']], outputCol='features'))


# In[15]:

# set up binomial logistic regression model with elastic net regularization.
lr = LogisticRegression(labelCol="label", featuresCol="features")


# In[16]:

# use pipeline to configure ML (first assemble features, then fit logistic regression)
pipeline = Pipeline(stages = [assembler, lr])


# In[17]:

############################################################################################
# Set up logistic regression parameter grid for cross validation (estimator parameter map) #
############################################################################################


# In[18]:

# maxIter indicates the maximum number of iteration for each model fitting
param_maxIter = [10,15,20]


# In[19]:

# regParam is lambda (>0) and elasticNetParam is alpha (0 <= alpha <= 1)
# alpha and lambda indicates L1 and L2 regularization
# when alpha = 1 --> lasso; when alpha = 0 --> ridge
# regularization prevents overfitting 
param_reg = [0.0, 0.05, 0.1]
param_elasticNet = [0.8, 0.85, 0.9]


# In[20]:

# set up the parameter grid
paramGrid = ParamGridBuilder().addGrid(lr.maxIter, param_maxIter).addGrid(lr.regParam, param_reg).addGrid(lr.elasticNetParam, param_elasticNet).build()


# In[21]:

###############################################
# Set up model evaluator for cross validation #
###############################################


# In[22]:

# set up model evaluator
evaluator = BinaryClassificationEvaluator()


# In[23]:

########################################################################
# Use k-fold cross validation to fit the dataset and find the best model
########################################################################


# In[24]:

# set number of folds for cross validation
k = 3


# In[25]:

# set up cross validation and fit the training dataset
crossval = CrossValidator(estimator = pipeline, estimatorParamMaps = paramGrid, evaluator = evaluator, numFolds = k)
cvModel = crossval.fit(df_train)


# In[26]:

#############################################
# Evaluate model result and generate output #
#############################################


# In[27]:

#########################
# Evaluate model result #
#########################


# In[28]:

# get the sample size
training_sample_size = df_train.count()
test_sample_size = df_test.count()
total_sample_size = df.count()


# In[29]:

# get average matrics for each configuration in cross validation, and the index of best model
avgMetrics = pd.DataFrame(cvModel.avgMetrics, columns = ['avgMetric'])
best_index = avgMetrics[avgMetrics['avgMetric'] == avgMetrics.max().iloc[0]].index[0]


# In[30]:

# find best model and summary
best = cvModel.bestModel.stages[1]
bestSum = best.summary


# In[31]:

# get the parameters of the best model
param = pd.Series(cvModel.getEstimatorParamMaps()[best_index])


# In[32]:

# get coefficients and odds ratio of each feature
coef = pd.DataFrame(data = best.coefficients.values,index = [x for x in df_train.columns if x not in ['label']], columns = ['Coefficient'])
coef['Odds_ratio'] = coef['Coefficient'].apply(np.exp)


# In[33]:

# get intercept
intercept = best.intercept


# In[34]:

# get number of classes(labels) and number of features
num_classes = best.numClasses
num_features = best.numFeatures


# In[35]:

# get full table of label, features, probability, and prediction
predictions = bestSum.predictions
display(predictions)


# In[36]:

# get area under ROC
AUROC = bestSum.areaUnderROC


# In[37]:

# convert prediciton and label column in predictions dataframe to RDD and instantiate metrics object
prediction_label = predictions.select('prediction', 'label').rdd
metrics = MulticlassMetrics(prediction_label)


# In[38]:

# get best model's F-1 score
f1_score = metrics.accuracy


# In[39]:

# get best model's precision of true and false
# precision of true
precision_T = metrics.precision(1)
# precision of false
precision_F = metrics.precision(0)


# In[40]:

# get best model's recall of true and false
# recall of true
recall_T = metrics.recall(1)
# recall of false
recall_F = metrics.recall(0)


# In[41]:

# get best model's confusion matrix
confusion_mtrx = metrics.confusionMatrix()


# In[42]:

# plot ROC curve
roc = bestSum.roc.toPandas()
lw = 2

trace_roc1 = go.Scatter(x=roc['FPR'], y=roc['TPR'], mode='lines', line=dict(color='darkorange', width=lw),name='ROC curve (area = %0.2f)' %bestSum.areaUnderROC)

trace_roc2 = go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='navy', width=lw, dash='dash'),showlegend=False)

layout_roc = go.Layout(title='Receiver operating characteristic',xaxis=dict(title='False Positive Rate'),yaxis=dict(title='True Positive Rate'))

fig_roc = plot(go.Figure(data=[trace_roc1, trace_roc2], layout=layout_roc), output_type='div')


# In[43]:

# plot precision against recall value
precision_recall = bestSum.pr.toPandas()

trace_pr = go.Scatter(x=precision_recall['recall'], y=precision_recall['precision'], mode='lines', line=dict(color='green', width=lw), name='precision recall curve')

layout_pr = go.Layout(title='Precision Recall',xaxis=dict(title='Recall'),yaxis=dict(title='Precision'))

fig_pr = plot(go.Figure(data=[trace_pr], layout=layout_pr), output_type='div')


# In[44]:

# plot F Measure, precision, and recall by threshold
fMeasure = bestSum.fMeasureByThreshold.toPandas()
precision_thre = bestSum.precisionByThreshold.toPandas()
recall_thre = bestSum.recallByThreshold.toPandas()

trace_fM_thre = go.Scatter(x=fMeasure['threshold'], y=fMeasure['F-Measure'], mode='lines', line=dict(color='green', width=lw), name='F Measure By Threshold Curve')

trace_pre_thre = go.Scatter(x=precision_thre['threshold'], y=precision_recall['precision'], mode='lines', line=dict(color='purple', width=lw), name='Precision By Threshold Curve')

trace_rec_thre = go.Scatter(x=recall_thre['threshold'], y=recall_thre['recall'], mode='lines', line=dict(color='orange', width=lw), name='Recall By Threshold Curve')

layout_fpr_thre = go.Layout(title='F-Measure, Precision, and Recall by Threshold',xaxis=dict(title='Threshold'),yaxis=dict(title='Value'))

fig_fpr_thre = plot(go.Figure(data=[trace_fM_thre, trace_pre_thre, trace_rec_thre], layout=layout_fpr_thre), output_type='div')


# In[45]:

# plot model performance lift curve
# step 1: get a dataframe of prediction probability against actual label
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
# build udf to find the last element
lastelement=udf(lambda v:float(v[-1]),DoubleType())
df_lift = predictions.withColumn('prob', lastelement(predictions.probability)).select('prob', 'label')
# step 2: quantileDiscretize the probability
# compute decile of probability
from pyspark.ml.feature import QuantileDiscretizer
discretizer = QuantileDiscretizer(numBuckets=10, inputCol="prob", outputCol="decile")
df_lift = discretizer.fit(df_lift).transform(df_lift)
# step 3: group dataframe by decile and compute average probability and label
df_lift = df_lift.groupby('decile').agg({'prob': 'mean', 'label': 'mean'}).toPandas()
df_lift.sort_values('decile', ascending = True, inplace = True)
df_lift['decile'] = df_lift['decile'].apply(lambda x: x+1)
# step 4: plot the lift curve
trace_lift1 = go.Scatter(x=df_lift['decile'], y=df_lift['avg(prob)'], mode='lines', line=dict(color='darkred', width=lw),name='Predicted Average Probability')

trace_lift2 = go.Scatter(x=df_lift['decile'], y=df_lift['avg(label)'], mode='lines', line=dict(color='navy', width=lw),name = 'Actual Average Label')

layout_lift = go.Layout(title='Predicted vs. Actual Lift Curve',xaxis=dict(title='Risk (Low -> High)'),yaxis=dict(title='Value'))

fig_lift = plot(go.Figure(data=[trace_lift1, trace_lift2], layout=layout_lift), output_type='div')


# In[46]:

##################################
# Output model evaluation result #
##################################


# In[47]:

# define a function to print all results
def print_result():
  print('\n Cross Validation Evaludation: ')
  print('\n Total Sample Size: %i' %total_sample_size)
  print('\n Training Sample Size: %i' %training_sample_size)
  print('\n Test Sample Size: %i' %test_sample_size)
  print('\n Average Metrics for Cross Validation: ')
  print(avgMetrics)
  print('\n ****************************************************')
  print('\n Best Logistic Regression Model Evaluation: ')
  print('\n Parameters of Best Model: ')
  print(param)
  print('\n Number of Label Classes: %i' %num_classes)
  print('\n Number of Features: %i' %num_features)
  print('\n Coefficients and Odds Ratio: ')
  print(coef)
  print('\n Intercept: %0.18f' %intercept)
  print('\n Area Under ROC: %0.18f' %AUROC)
  print('\n F-1 Score: %0.18f' %f1_score)
  print('\n Precision of True: %0.18f' %precision_T)
  print('\n Precision of False: %0.18f' %precision_F)
  print('\n Recall of True: %0.18f' %recall_T)
  print('\n Recall of False: %0.18f' %recall_F)
  print('\n Confusion Matrix: ')
  print(confusion_mtrx.toArray())


# In[48]:

# print evaluation result
print_result()


# In[49]:

# display ROC curve
displayHTML(fig_roc)


# In[50]:

# display precision_recall curve
displayHTML(fig_pr)


# In[51]:

# display F Measure, precision, and recall by threshold graph
displayHTML(fig_fpr_thre)


# In[52]:

# display performance lift curve
displayHTML(fig_lift)


# In[53]:



