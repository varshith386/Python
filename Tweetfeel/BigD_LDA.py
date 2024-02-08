#Working

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, CountVectorizer, VectorAssembler
from pyspark.ml.clustering import LDA

# Create a Spark session
spark = SparkSession.builder.appName("Spark").getOrCreate()

# Load data into a Spark DataFrame
df = spark.read.csv("file:///C:/Users/hp/OneDrive/Desktop/Pyhton/test.csv", header=True, inferSchema=True)

# Tokenize the 'text' column
tokenizer = Tokenizer(inputCol="text", outputCol="words")
df = tokenizer.transform(df)

# Use CountVectorizer to convert the words to feature vectors
cv = CountVectorizer(inputCol="words", outputCol="rawFeatures")
model_cv = cv.fit(df)
df = model_cv.transform(df)

# Assemble features into a single vector column
feature_cols = ["rawFeatures"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="feature_vector")
df = assembler.transform(df)

# Split the data into training and testing sets
train_df, test_df = df.randomSplit([0.67, 0.33], seed=42)

# LDA Model
lda = LDA(k=5, featuresCol="feature_vector", maxIter=10)

# Fit the model
model = lda.fit(train_df)

# Make predictions on the test set
predictions = model.transform(test_df)

# Perplexity
perplexity = model.logPerplexity(test_df)
print(f"Perplexity: {perplexity:.2f}")

spark.stop()
