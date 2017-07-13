# testing pyspark dataframe functions and data manipulation 
# code is executed on Databricks

import pyspark
import numpy as np
import matplotlib.pyplot as plt

# import dataframe from s3
filepath = 's3a://mdv-sandbox/michael/python_testing/data/raw'
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




