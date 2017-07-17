# PySpark features extraction, transformation and selection testing
# code will be executed on databricks, so there is no need to setup spark environment

import pyspark

# import dataframe from s3
filepath = 's3a://somefilepathhere'
df = spark.read.parquet(filepath)
df.printSchema()

# features and corresponding datatypes: 
# 	numeric, binary, ordinal (3 levels high, medium, low, and another for null values)
# target lable: positive and negative (1 and 0)

# transform binary features
# if there is more than 1 stage of data transformation, 
# use pipeline to tie these stages together, which simplifies the code

# 1. transform binary or categorical features
# 1.1 use StringIndexer
# 	the most frequent label has lower index value
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol = 'patient_gender', outputCol = 'gender_index')
df_indexed = indexer.fit(df).transform(df)
df_indexed.show()

# 1.2 use OneHotEncoder
# 	map a column of label indices to a column of binary vectors, with at most one single value
from pyspark.ml.feature import OneHotEncoder, StringIndexer

# first get label indices using StringIndexer
indexer = StringIndexer(inputCol = 'patient_gender', outputCol = 'gender_index')
df_indexed = indexer.fit(df).transform(df)
df_indexed.show()

# then use OneHotEncoder to map the indices
encoder = OneHotEncoder(includeFirst = False, inputCol = 'gender_index', outputCol = 'gender_vec')
df_encoded = encoder.transform(df_indexed)

# 1.3 use udf (user defined function)
# if the catetgorical variables are known and the number of levels is small, use udf to convert
from pyspark.sql.functions import udf
def gender_map(gender):
	if gender == 'Male': return 1
	elif gender == 'Female': return 2
	else : return 0
udfGenderToNumber= udf(gender_map,IntegerType())
df.withColumn('patient_gender_index', udfGenderToNumber('patient_gender')).show()

# the udf is also applicable for ordinal variables, if the number of levels are limited
# and the order of levels are known

# 1.4 use VectorIndexer
# 	automatically decide which features are categorical and convert original values to category indices.
# 	input: 
# 		inputCol, outputCol, MaxCategories
# 	decide which feature will be categorical based on number of distinct values, features less than
#	MaxCategories will be declared as categorical.
# 	primarily for classification trees. 
from pyspark.ml.feature import VectorIndexer
indexer = VectorIndexer(inputCol = 'patient_gender', outputCol = 'gender_indexed', maxCategories = 10)
indexerModel = indexer.fit(df)
indexedDf = indexerModel.transform(df)
