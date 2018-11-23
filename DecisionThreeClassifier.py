import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

print("\nStart")

df = pd.read_csv("train.csv").as_matrix()

x = df[0:, 1:]
y = df[0:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y)

print(len(x_train))
print(len(y_train))

print(len(x_test))
print(len(y_test))

clf = DecisionTreeClassifier()

clf.fit(x_train[:50], y_train[:50])

predictions = clf.predict(x_test)

print("\nConfusion matrix: ")
print(confusion_matrix(y_test, predictions))

print("\nClassification report: ")
print(classification_report(y_test, predictions))

print("\nAccuracy score: ")
print(accuracy_score(y_test, predictions))

print("\n\n\n\n")
print("class\tprecision")
matrix = classification_report(y_test, predictions)

lines = matrix.split("\n")
# https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format
for m in lines[2: -3]:
    row = {}
    row_data = m.split('      ')
    # row_data = list(filter(None, row_data))
    row['class'] = row_data[1]
    row['precision'] = float(row_data[2])
    row['recall'] = float(row_data[2])
    row['f1_score'] = float(row_data[3])
    row['support'] = float(row_data[4])
    print(row['class'], end="\t")
    print(row['precision'])
    # report_data.append(row)

print("\nStop")
