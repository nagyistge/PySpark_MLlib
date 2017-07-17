# testing pyspark dataframe functions and data manipulation 
# code is executed on Databricks

import pyspark
from pyspark import SQLContext
from pyspark import SparkContext
sc = SparkContext()

# import dataframe from s3
filepath = 's3a://somefilepathhere'
df = spark.read.parquet(filepath)
df.printSchema()

# count size of dataframe
df.count()

# show first n observations
n = 5 
df.head(n)

# show column number and column names
len(df.columns)
df.columns

# show summary statistics of numerical columns in dataframe
df.describe().show()

# select columns in dataframe
df.select('patient_date_of_birth', 'patient_gender', 'token4').show()

# number of distinct entries in a columns (similar to nunique)
df.select('patient_gender').distinct().count()

# calculate pariwise frequency of categorical columns using crosstab
df.crosstab('patient_date_of_birth', 'patient_gender').show()

# drop duplicate rows
df.select('token4', 'patient_date_of_birth').dropDuplicates().show()

# drop rows with null value
df.dropna().count()

# fill null values in a column with certain value
df.select('patient_gender').fillna('Unknown').show()

# filter rows using a condition
df.filter(df.patient_gender == 'Unknown').count()

# groupby and find mean of certain columns
df.groupby('token4').agg({'patient_date_of_birth': 'mean'}).show()

# groupby and count number of entries
df.groupby('patient_date_of_birth').count().show()

# create sample dataframe 
# 3 parameters: 
#			1. withReplacement = True or False to select a observation with or without replacement.
#			2. fraction = x, where x = .5 shows that we want to have 50% data in sample DataFrame.
#			3. seed for reproduce the result
train1 = df.sample(False, 0.3, 1) 

# apply map functions on columns of dataframe
# print first 5 elements of mapped year in date of birth
df.select('patient_date_of_birth').map(lambda x: str(x)[:5]).take(5)

# sort dataframe based on columns
df.orderBy(df.patient_date_of_birth.desc()).show(5)

# add new columns in dataframe
# add a patient age column
df.withColumn('patient_age', 2017 - df.patient_date_of_birth.str()[:5].astype(int)).select('patient_date_of_birth', 'patient_age').show(5)

# drop columns in dataframe
df.drop('token9').columns

# find difference token4 between two dataframes (a super set and a subset)
diff_df = df.select('token4').subtract(train1.select('token4'))
diff_df.distinct().count()

# apply SQL queries on a dataframe
# 1. register dataframe as a table
df.registerAsTable('df_table')

# 2. apply SQL queries on table, result is a dataframe
sqlContext.sql('select patient_age, count(token4) from df_table group by patient_age').show()
