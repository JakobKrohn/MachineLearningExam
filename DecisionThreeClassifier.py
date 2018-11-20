import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print("\nStart")

df = pd.read_csv("train.csv").as_matrix()

# print(df)

# Take all rows, all columns except first
x = df[0:, 1:]
y = df[0:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y)

print(len(x_train))
print(len(y_train))

print(len(x_test))
print(len(y_test))

clf = DecisionTreeClassifier()

clf.fit(x_train, y_train)

predictions = clf.predict(x_test)

print("\nConfusion matrix: ")
print(confusion_matrix(y_test, predictions))

print("\nCassification report: ")
print(classification_report(y_test, predictions))

print("\nStop")
