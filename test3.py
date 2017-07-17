# PySpark Logsitic Regression testing
# code executed on databricks

import pyspark
from pyspark import SQLContext
from pyspark import SparkContext
sc = SparkContext()

# import dataframe from s3, which will be used as a training dataset
filepath = 's3a://somefilepathhere'
df = spark.read.parquet(filepath)
df.printSchema()

# set up binomial logistic regression model with elastic net regularization.
# regParam is lambda (>0) and elasticNetParam is alpha (0 <= alpha <= 1)
# alpha and lambda indicates L1 and L2 regularization
# when alpha = 1 --> lasso; when alpha = 0 --> ridge
# regularization prevents overfitting 
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(maxIter = 10, regParam = 0.3, elasticNetParam = 0.8)

# fit the model
lrModel = lr.fit(df)

# print coefficients and intercept
print('Coefficient: ', str(lrModel.coefficients))
print('Intercept: ', str(lrModel.intercept))

# Another way is to use multinomial logistic regression and 
# specifying family as 'multinomial' for binary classification

# set up multinomial logistic regression model with regularization
mlr = LogisticRegression(maxIter = 10, regParam = 0.3, elasticNetParam = 0.8, family = 'multinomial')

# fit the model
mlrModel = mlr.fit(df)

# print coefficients and intercept
print('Coefficient: ', str(mlrModel.coefficients))
print('Intercept: ', str(mlrModel.intercept))

# Extract the summary from the returned LogisticRegression model instance trained in training df
trainingSummary = lrModel.summary

# obtain the objective per iteration
objectiveHistory = trainingSummary.objectiveHistory
print('objectiveHistory')
for i in objectiveHistory:
	print(i)

# obtain the receiver-operating characteristic as a dataframe and areaunderROC
trainingSummary.roc.show()
print('Area Under ROC: ', str(trainingSummary.areaunderROC))

# set the model threshold to maximize F-measure
fMeasure = trainingSummary.fMeasureByThreshold
maxFMeasure = fMeasure.groupby().max('F-Measure').select('max(F-Measure)').head()
bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure']).select('threshold').head()['threshold']
lr.setThreshold(bestThreshold)



