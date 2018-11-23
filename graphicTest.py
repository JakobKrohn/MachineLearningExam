import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


def get_text(name, value):

    text = name + ": "
    text += str(value)
    return text


def get_color(actual, predicted):

    if actual != predicted:
        return "red"
    return "green"


def plot_images(mlp_prediction, gnb_prediction, dtc_prediction, knn_prediction, svm_prediction):

    print("\nPlotting images")

    d = x_test[0]
    d.shape = (28, 28)

    columns = 8
    rows = 10

    plt.rcParams["figure.figsize"] = [15, 9]
    fig = plt.figure()

    for i in range(1, columns * rows + 1):
        figure = fig.add_subplot(rows, columns, i)
        label = y_test[i]
        print("# %d = %d" % (i, label))
        img = x_test[i]

        img.shape = (28, 28)
        figure.imshow(255 - img)

        figure.axis('off')
        figure.set_title(label)

        figure.text((rows + 30), columns - 10, get_text("MLP", mlp_prediction[i]), color=get_color(label, mlp_prediction[i]))
        figure.text((rows + 30), (columns + 0), get_text("KNN", knn_prediction[i]), color=get_color(label, knn_prediction[i]))
        figure.text((rows + 30), (columns + 10), get_text("DTC", dtc_prediction[i]), color=get_color(label, dtc_prediction[i]))
        figure.text((rows + 30), (columns + 20), get_text("GNB", gnb_prediction[i]), color=get_color(label, gnb_prediction[i]))
        figure.text((rows + 30), (columns + 30), get_text("SVM", svm_prediction[i]), color=get_color(label, svm_prediction[i]))

    plt.show()


df = pd.read_csv("train.csv").as_matrix()

# df = df[:500]

print(df.shape)

x = df[0:, 1:]
y = df[0:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y)

print("Testing set size: %d" % len(x_test))

# MLP CLASSIFIER
print("\nTraining MLP classifier")
mlp = MLPClassifier(verbose=True, solver='adam', activation='relu', learning_rate='constant', hidden_layer_sizes=(783,))
# mlp.fit(x_train[:10], y_train[:10])
mlp.fit(x_train, y_train)
print("\nTesting MLP classifier")
mlp_pred = mlp.predict(x_test)
mlp_accuracy = accuracy_score(y_test, mlp_pred)
mlp_report = classification_report(y_test, mlp_pred)
print("Accuracy score: %f" % accuracy_score(y_test, mlp_pred))
print(mlp_report)
# mlp_predictions = mlp_pred[:26]
mlp_predictions = mlp_pred[:81]

# K NEAREST NEIGHBORS
print("\nTraining k nearest neighbors")
knn = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
knn.fit(x_train, y_train)
print("\nTesting KNN")
knn_pred = knn.predict(x_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
knn_report = classification_report(y_test, knn_pred)
print("Accuracy score: %f" % accuracy_score(y_test, knn_pred))
print(knn_report)
knn_predictions = knn_pred[:81]

# DECISION THREE CLASSIFIER
print("\nTraining Decision Three Classifier")
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
print("\nTesting Decision Three Classifier")
dtc_pred = dtc.predict(x_test)
dtc_accuracy = accuracy_score(y_test, dtc_pred)
dtc_report = classification_report(y_test, dtc_pred)
print("Accuracy score: %f" % accuracy_score(y_test, dtc_pred))
print(dtc_report)
dtc_predictions = dtc_pred[:81]

# GAUSSIAN NAIVE BAYES
print("\nTraining Gaussian Naive Bayes")
gnb = GaussianNB()
gnb.fit(x_train, y_train)
print("\nTesting Gaussian Naive Bayes")
gnb_pred = gnb.predict(x_test)
gnb_accuracy = accuracy_score(y_test, gnb_pred)
gnb_report = classification_report(y_test, gnb_pred)
print("Accuracy score: %f" % accuracy_score(y_test, gnb_pred))
print(gnb_report)
gnb_predictions = gnb_pred[:81]

# SUPPORT VECTOR MACHINE
print("\nTraining Support Vector Machine")
svm = SVC(kernel="poly", degree=3)
svm.fit(x_train, y_train)
print("\nTesting Support Vector Machine")
svm_pred = svm.predict(x_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_report = classification_report(y_test, svm_pred)
print("Accuracy score: %f" % accuracy_score(y_test, svm_pred))
print(svm_report)
svm_predictions = svm_pred[:81]

# PLOT IMAGES AND PREDICTIONS
plot_images(mlp_predictions, knn_predictions, dtc_predictions, gnb_predictions, svm_predictions)

# Classification split
# https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format

# PLOT BAR GRAPHS
# https://matplotlib.org/gallery/statistics/barchart_demo.html
print("\nPlotting bar graphs")
n_groups = 10
bar_width = 0.15
index = np.arange(n_groups)

plt.rcParams["figure.figsize"] = [16, 9]
fig, ax = plt.subplots()

# Get MLP
mlp_precision = []
lines = mlp_report.split("\n")
for m in lines[2: -3]:
    row = {}
    row_data = m.split('      ')
    row['class'] = row_data[1]
    row['precision'] = float(row_data[2])
    # print(row['class'], end="\t")
    # print(row['precision'])
    mlp_precision.append(row['precision'])

# Get KNN
knn_precision = []
lines = knn_report.split("\n")
for m in lines[2: -3]:
    row = {}
    row_data = m.split('      ')
    row['class'] = row_data[1]
    row['precision'] = float(row_data[2])
    knn_precision.append(row['precision'])

# Get DTC
dtc_precision = []
lines = dtc_report.split("\n")
for m in lines[2: -3]:
    row = {}
    row_data = m.split('      ')
    row['class'] = row_data[1]
    row['precision'] = float(row_data[2])
    dtc_precision.append(row['precision'])

# Get GNB
gnb_precision = []
lines = gnb_report.split("\n")
for m in lines[2: -3]:
    row = {}
    row_data = m.split('      ')
    row['class'] = row_data[1]
    row['precision'] = float(row_data[2])
    gnb_precision.append(row['precision'])

# Get SVM
svm_precision = []
lines = svm_report.split("\n")
for m in lines[2: -3]:
    row = {}
    row_data = m.split('      ')
    row['class'] = row_data[1]
    row['precision'] = float(row_data[2])
    svm_precision.append(row['precision'])

rect_mlp = ax.bar(index - (bar_width * 1.5), mlp_precision, bar_width, label="MLP")

rect_knn = ax.bar(index - (bar_width * 0.5), knn_precision, bar_width, label="KNN")

rect_dtc = ax.bar(index + (bar_width * 0.5), dtc_precision, bar_width, label="DTC")

rect_gnb = ax.bar(index + (bar_width * 1.5), gnb_precision, bar_width, label="GNB")

rect_svm = ax.bar(index + (bar_width * 2.5), svm_precision, bar_width, label="SVM")

ax.set_xlabel("X label")
ax.set_ylabel("Y label")
ax.set_title("Precision for each number (label)")
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
# ax.tick_params(length=9)
ax.legend()

plt.show()

# PLOT PIE CHART
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.bar.html

labels = 'MLP', 'KNN', 'DTC', 'GNB', 'SVM'
sizes = [mlp_accuracy, knn_accuracy, dtc_accuracy, gnb_accuracy, svm_accuracy]

fig, ax = plt.subplots()


# https://stackoverflow.com/questions/6170246/how-do-i-use-matplotlib-autopct
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = pct * total
        return '{p:.1f}%'.format(p=val)
    return my_autopct


# ax.pie(sizes, labels=labels, autopct='%1.1f%%')
ax.pie(sizes, labels=labels, autopct=make_autopct(sizes))
ax.axis('equal')
plt.title("Accuracy score")
plt.legend()
plt.show()

