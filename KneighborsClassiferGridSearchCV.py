import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#Need this if statement if you have windows.
if __name__ == "__main__":
    df = pd.read_csv("train.csv").as_matrix()

    print(df.shape)

    x = df[0:, 1:]
    y = df[0:, 0]

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    neigh = KNeighborsClassifier()

    ns = np.arange(3, 30, 1)
    ns = list(ns)
    print(ns)

    '''
    parameter_space = {
        'n_neighbors': ns,
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': list(np.arange(5, 50, 1)),
        'p': [1, 2, 3],
    }'''

    parameter_space = {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        'algorithm': ['brute', 'ball_tree', 'kd_tree'],
    }

    from sklearn.model_selection import GridSearchCV

    # clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3, verbose=10)
    clf = GridSearchCV(neigh, parameter_space, n_jobs=2, cv=3, verbose=10)
    clf.fit(x_train[:5000], y_train[:5000])

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