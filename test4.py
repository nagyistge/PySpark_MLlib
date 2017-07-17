# pyspark logistic regression model tuning
# code executed on databricks

import pyspark
from pyspark import SQLContext
from pyspark import SparkContext
from pyspark.ml import Pipeline 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

sc = SparkContext()

# import dataframe from s3 (training dataset)
filepath = 's3a://somefilepathhere'
df = spark.read.parquet(filepath)
df.printSchema()

# import dataframe from s3 (test dataset)
filepath_t = 's3a://somefilepathhere'
df_t = spark.read.parquet(filepath)
df_t.printSchema()

# MLlib supports k-fold cross validation and train validation split

# required input:
# 	estimator: algorithm or pipeline to tune
# 	set of paramMaps: parameters to choose from (parameter grid)
# 	evaluator: metric to measure how well a fitted model performs on tested data

# workflow:
# 	split data into training and validation subsets
# 	for each (train,test), iterate through the set of paramMaps
# 		for each paramMap, fit the estimator using those parameters, get the fitted model, 
# 		then evaluate model performance using evaluator
# 	select model produced by best-performing set of parameters

# 1. k-fold cross validation
# split dataset into a set of folds (seperate training [(k-1)/k] and testing [1/k] datasets) 
# use pipeline to stage the order of process, convert categorical variable first, then fit model

# 1.1 use VectorIndexer to convert categorical value
from pyspark.ml.feature import VectorIndexer
indexer = VectorIndexer(inputCol = 'patient_gender', outputCol = 'gender_indexed')

# 1.2 intiate logistic regression model
lr = LogisticRegression(maxIter = 10)

# 1.3 use pipeline to configure ML
pipeline = Pipeline(stages = [indexer, lr])

# 1.4 treat pipeline as an extimator, wrap it in a CrossValidator instance
# 	choose parameters for each stage, use ParamGridBuilder 
# 	with 3 values for idnexer and 2 values for lr, we have 6 parameter settings
paramGrid = ParamGridBuilder().addGrid(indexer.maxCategories, [10,20,30]).addGrid(lr.regParam, [0.2,0.3]).build()
crossval = CrossValidator(estimator = pipeline, estimatorParamMaps = paramGrid, evaluator = BinaryClassificationEvaluator, numFolds = 3)

# 1.5 run cross validation, choose the best set of parameters
cvModel = crossval.fit(df)

# 1.6 predict on test dataset, cvModel uses the best model found (lrModel)
prediction = cvModel.transform(df_t)
selected = prediction.select('token4', 'patient_gender', 'patient_date_of_birth')
for i in selected.collect():
	print(i)

# 2. train validation split
# only evaluate each combination of parameters once, as opposed to k-fold cross validation
# train validation split creates a single (training, test) pair using trainRatio parameter
from pyspark.ml.tuning import TrainValidationSplit

# 2.1 prepare training, test pair
df_train, df_test = df.randomSplit([0.9,0.1], seed = 123)

# 2.2 set up logistic regression model
lr = LogisticRegression(maxIter = 10)

# 2.3 configure paramGridBuilder
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.2,0.3]).addGrid(elasticNetParam, [0.7,0.8,0.85]).build()


# 2.4 the estimator is logistic regression instead of pipeline
# 70% of data will be used for training, 30% for validation
tvs = TrainValidationSplit(extimator = lr, estimatorParamMaps =paramGrid, evaluator = BinaryClassificationEvaluator(), trainRatio = 0.7)

# 2.5 run tvs, and choose best set of parameters
model = tvs.fit(df_train)

# 2.6 predict on test dataset, model uses the best set of parameters
model.transform(df_test).select('token4', 'patient_gender', 'patient_date_of_birth').show()

