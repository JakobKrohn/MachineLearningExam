# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html


import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

df = pd.read_csv("train.csv").as_matrix()

print(df.shape)

x = df[0:, 1:]
y = df[0:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y)

# mlp = MLPClassifier(hidden_layer_sizes=(90, 90, 90), verbose=True)


# as 0.9518095238095238
# mlp = MLPClassifier(hidden_layer_sizes=(783, 783, 783), verbose=True)

# as 0.9575238095238096
# mlp = MLPClassifier(hidden_layer_sizes=(783, 783, 783), verbose=True, solver='sgd')

mlp = MLPClassifier(verbose=True, solver='adam', activation='relu', learning_rate='constant', hidden_layer_sizes=(783,))

mlp.fit(x_train, y_train)

predictions = mlp.predict(x_test)

print("\nConfusion matrix: ")
print(confusion_matrix(y_test, predictions))

print("\nClassification report: ")
print(classification_report(y_test, predictions))

print("\nAccuracy score: ")
print(accuracy_score(y_test, predictions))

print("Training set score: %f" % mlp.score(x_train, y_train))
print("Test set score: %f" % mlp.score(x_test, y_test))
