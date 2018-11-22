import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score


def fitModel(model):

    model.fit(x_train, y_train)


print("start\n\n")

df = pd.read_csv("train.csv").as_matrix()

x = df[0:, 1:]
y = df[0:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y)

models = (
    MLPClassifier(verbose=True, solver="adam"),
    MLPClassifier(verbose=True, solver="sgd"),
    # MLPClassifier(hidden_layer_sizes=(783,), verbose=True, solver='sgd', shuffle=True),
    # MLPClassifier(hidden_layer_sizes=(783, 783, 783), verbose=True, solver='adam', shuffle=True),
)

titles = (
    "",
    "",
    "",
)

print("[Fitting models]")

for model in models:

    print("\n")
    print(model)

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    print(classification_report(y_test, predictions))

    print("\nAccuracy score: %s" % accuracy_score(y_test, predictions))


print("\n[Testing models]")

for model in models:

    print("\n")
    print(model)

    predictions = model.predict(x_test)

    print(classification_report(y_test, predictions))

    print("\nAccuracy score: %s" % accuracy_score(y_test, predictions))

print("\n\nstop")
