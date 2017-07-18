# pyspark model evaluation and plotting
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

# split dataframe into training and test set
training, test = df.randomSplit([0.7,0.3], seed = 1234)

# train a logistic regression model
lr = LogisticRegression(maxIter = 10, regParam = 0.3, elasticNetParam = 0.8)
lrModel = lr.fit(training)

#