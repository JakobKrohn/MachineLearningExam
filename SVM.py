from sklearn import svm
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

print("\nStart")

if __name__ == "__main__":
    df = pd.read_csv("train.csv").as_matrix()
    x = df[0:, 1:]
    y = df[0:, 0]

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    parameter_space = {
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': [0.0001, 0.01, 0.0005],
        }
    clf = GridSearchCV(svm.SVC(), parameter_space, n_jobs=2, cv=3, verbose=10)
    clf.fit(x_train[:5000], y_train[:5000])
    predictions = clf.predict(x_test)

    # Best paramete set
    print('Best parameters found:\n', clf.best_params_)

    # All results
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    print("\nConfusion matrix: ")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report: ")
    print(classification_report(y_test, predictions))

    print("\nAccuracy score: ")
    print(accuracy_score(y_test, predictions))