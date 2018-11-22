import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv("train.csv").as_matrix()

x = df[0:, 1:]
y = df[0:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(verbose=True)

alp = np.arange(0.0001, 0.1, 0.0001)
alp = list(alp)
print(alp)

parameter_space = {
    # 'hidden_layer_sizes': [(100, 100, 100), (783, 783, 783), (783,)],
    'activation': ['relu'],
    'solver': ['adam'],
    'learning_rate': ['constant'],
}

from sklearn.model_selection import GridSearchCV

# clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3, verbose=10)
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3, verbose=10)
clf.fit(x_train, y_train)

# Best paramete set
print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

from sklearn.metrics import classification_report
y_pred = clf.predict(x_test)
print('Results on the test set:')
print(classification_report(y_test, y_pred))

