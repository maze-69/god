

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

df=pd.read_csv('/content/diabetes.csv')

df

df.columns

df.info(verbose=True)

df.describe()

"""Check for null values. If present remove null values from the dataset"""

df.isnull().sum()

"""Outcome is the label/target, other columns are features"""

X = df.drop('Outcome',axis = 1)
y = df['Outcome']

from sklearn.preprocessing import scale
X = scale(X)
# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 14)

X_train.shape

y_train.shape

X_test.shape

y_test.shape

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
 
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

#  The confusion matrix is used to evaluate the accuracy of classification. 
# The first column and first row represent True Positive, the second column and 
# first row represent False Positive, the second row and first column represent False Negative, 
# and the second row and second column represent True Negative.

print("Confusion matrix: ")
cs = metrics.confusion_matrix(y_test,y_pred)
print(cs)

pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

print("Accuracy ",metrics.accuracy_score(y_test,y_pred))

# 137+35= 172/231= 74.45%

# Accuracy = 1 – Misclassification rate    1 - 0.2554 = 0.7446

# Accuracy (ACC) is calculated as the number of all correct predictions divided by the total number of the dataset.

# Accuracy = 1 – Error Rate. 

# The best accuracy is 1.0, whereas the worst is 0.0.

# Error Rate = 1 – Accuracy

# Outcome takes values 0 or 1 where 0 indicates negative for diabetes and 1 indicates positive for diabetes.

# Error rate (ERR) is calculated as the number of all incorrect predictions divided by the total number of the dataset.(positive+negative)

# The best error rate is 0.0, whereas the worst is 1.0.

# Misclassification Rate =  incorrect predictions / total predictions.
# 59 / 231 = 0.25 

# Misclassification Rate = (false positive + false negative) / (total predictions)
# 20 + 39 = 59/231 = 0.25 

total_misclassified = cs[0,1] + cs[1,0]
print(total_misclassified)

total_examples = cs[0,0]+cs[0,1]+cs[1,0]+cs[1,1]
print(total_examples)

print("Error rate",total_examples)
print("Error rate ",1-metrics.accuracy_score(y_test,y_pred))

# Precision Score

# TP – True Positives
# FP – False Positives
# Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. 
# Precision – Accuracy of positive predictions.
# Precision = TP/(TP + FP)

print("Precision score",metrics.precision_score(y_test,y_pred))

# Recall Score
# FN – False Negatives

# Recall(sensitivity or true positive rate): Fraction of positives that were correctly identified.
# Recall (Sensitivity) - Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes. 
# Recall = TP/(TP+FN)

# Sensitivity: The “true positive rate” – the percentage of positive outcomes the model is able to detect.
# Specificity: The “true negative rate” – the percentage of negative outcomes the model is able to detect.

print("Recall score ",metrics.recall_score(y_test,y_pred))

print("Classification report ",metrics.classification_report(y_test,y_pred))
# F1 Score- A helpful metric for comparing two classifier
# F1 Score: A metric that tells us the accuracy of a model, relative to how the data is distributed.
# It is created by finding the the harmonic mean of precision and recall.
# F1 = 2 * (precision * recall)/(precision + recall)
