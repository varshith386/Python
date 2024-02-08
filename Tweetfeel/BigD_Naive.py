#Working
#Plotting of all classes is in this py file

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
#x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state = 42)
from sklearn.preprocessing import StandardScaler

from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import NaiveBayes, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import classification_report





spark = SparkSession.builder.appName("Spark").config("spark.driver.memory", "4g").getOrCreate()
df = spark.read.csv("file:///C:/Users/hp/OneDrive/Desktop/Pyhton/test.csv", header=True, inferSchema=True)




tokenizer = Tokenizer(inputCol="text", outputCol="words")
df = tokenizer.transform(df)
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
df = hashingTF.transform(df)


idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(df)
df = idfModel.transform(df)

train_df, test_df = df.randomSplit([0.67, 0.33], seed=42)


nb = NaiveBayes(featuresCol="features", labelCol="label", smoothing=1.0)
nb_model = nb.fit(train_df)

predictions = nb_model.transform(test_df)

y_test = predictions.select("label").collect()
y_pred = predictions.select("prediction").collect()
y_test = [int(row.label) for row in y_test]
y_pred = [row.prediction for row in y_pred]

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
overall_accuracy = evaluator.evaluate(predictions)
print(f"Overall Accuracy: {overall_accuracy:.2f}")

evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = evaluator_f1.evaluate(predictions)
print(f"F1 Score: {f1_score:.2f}")


report = classification_report(y_test, y_pred, output_dict=True)


for class_label, metrics in report.items():
    if class_label.isdigit():
        accuracy = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1-score']

        print(f"Class {class_label} Metrics:")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print()
spark.stop()


df = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\Pyhton\test.csv")
label_mapping = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
df['emotion'] = df['label'].map(label_mapping)


count = df['emotion'].value_counts()

plt.figure(figsize=(8, 8))
count.plot(kind='pie', autopct='%1.1f%%', startangle=90, explode=[0.1] * len(count))
plt.ylabel('')
plt.show()