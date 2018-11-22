import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

df = pd.read_csv("train.csv").as_matrix()

x = df[0:, 1:]
y = df[0:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y)

p = Perceptron(random_state=42, max_iter=10)
p.fit(x_train, y_train)

predictions = p.predict(x_test)

print("\nConfusion matrix: ")
print(confusion_matrix(y_test, predictions))

print("\nClassification report: ")
print(classification_report(y_test, predictions))

print("\nAccuracy score: ")
print(accuracy_score(y_test, predictions))

