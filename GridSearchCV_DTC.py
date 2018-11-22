import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


df = pd.read_csv("train.csv").as_matrix()

print(df.shape)

x = df[0:, 1:]
y = df[0:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y)

parameter_space = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
}

clf = DecisionTreeClassifier()

gscv = GridSearchCV(clf, parameter_space, n_jobs=3, cv=3, verbose=10)

gscv.fit(x_train, y_train)

# Best paramete set
print('Best parameters found:\n', gscv.best_params_)

# All results
means = gscv.cv_results_['mean_test_score']
stds = gscv.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, gscv.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

from sklearn.metrics import classification_report
y_pred = gscv.predict(x_test)
print('Results on the test set:')
print(classification_report(y_test, y_pred))

'''
Best parameters found:
 {'criterion': 'entropy', 'splitter': 'best'}
0.837 (+/-0.005) for {'criterion': 'gini', 'splitter': 'best'}
0.836 (+/-0.002) for {'criterion': 'gini', 'splitter': 'random'}
0.847 (+/-0.007) for {'criterion': 'entropy', 'splitter': 'best'}
0.841 (+/-0.007) for {'criterion': 'entropy', 'splitter': 'random'}
Results on the test set:
             precision    recall  f1-score   support

          0       0.92      0.92      0.92      1002
          1       0.95      0.95      0.95      1149
          2       0.83      0.83      0.83      1045
          3       0.84      0.84      0.84      1112
          4       0.85      0.86      0.86      1025
          5       0.81      0.82      0.81       949
          6       0.89      0.89      0.89      1044
          7       0.89      0.91      0.90      1082
          8       0.82      0.78      0.80      1022
          9       0.83      0.82      0.82      1070

avg / total       0.86      0.86      0.86     10500
'''