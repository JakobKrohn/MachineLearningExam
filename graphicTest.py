import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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

# 5
# 5

    columns = 8
    rows = 10

    # plt.rcParams["figure.figsize"] = [14, 6]
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

print(df.shape)

x = df[0:, 1:]
y = df[0:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

print("Testing set size: %d" % len(x_test))

# x_train = x_train[:10]
# y_train = y_train[:10]

# MLP CLASSIFIER
print("\nTraining MLP classifier")
mlp = MLPClassifier(verbose=True, solver='adam', activation='relu', learning_rate='constant', hidden_layer_sizes=(783,))
# mlp.fit(x_train[:10], y_train[:10])
mlp.fit(x_train, y_train)
print("\nTesting MLP classifier")
mlp_pred = mlp.predict(x_test)
mlp_accuracy = accuracy_score(y_test, mlp_pred)
print("Accuracy score: %f" % accuracy_score(y_test, mlp_pred))
print("Training set score: %f" % mlp.score(x_train, y_train))
print("Test set score: %f" % mlp.score(x_test, y_test))
# mlp_predictions = mlp_pred[:26]
mlp_predictions = mlp_pred[:81]

# K NEAREST NEIGHBORS
print("\nTraining k nearest neighbors")
knn = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
knn.fit(x_train, y_train)
print("\nTesting KNN")
knn_pred = knn.predict(x_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
print("Accuracy score: %f" % accuracy_score(y_test, knn_pred))
print("Training set score: %f" % knn.score(x_train, y_train))
print("Test set score: %f" % knn.score(x_test, y_test))
knn_predictions = knn_pred[:81]

# DECISION THREE CLASSIFIER
print("\nTraining Decision Three Classifier")
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
print("\nTesting Decision Three Classifier")
dtc_pred = dtc.predict(x_test)
dtc_accuracy = accuracy_score(y_test, dtc_pred)
print("Accuracy score: %f" % accuracy_score(y_test, dtc_pred))
print("Training set score: %f" % dtc.score(x_train, y_train))
print("Test set score: %f" % dtc.score(x_test, y_test))
dtc_predictions = dtc_pred[:81]

# GAUSSIAN NAIVE BAYES
print("\nTraining Gaussian Naive Bayes")
gnb = GaussianNB()
gnb.fit(x_train, y_train)
print("\nTesting Gaussian Naive Bayes")
gnb_pred = gnb.predict(x_test)
gnb_accuracy = accuracy_score(y_test, gnb_pred)
print("Accuracy score: %f" % accuracy_score(y_test, gnb_pred))
print("Training set score: %f" % gnb.score(x_train, y_train))
print("Test set score: %f" % gnb.score(x_test, y_test))
gnb_predictions = gnb_pred[:81]

# SUPPORT VECTOR MACHINE
print("\nTraining Support Vector Machine")
svm = SVC(kernel="poly", degree=3)
svm.fit(x_train, y_train)
print("\nTesting Support Vector Machine")
svm_pred = svm.predict(x_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
print("Accuracy score: %f" % accuracy_score(y_test, svm_pred))
print("Training set score: %f" % svm.score(x_train, y_train))
print("Test set score: %f" % svm.score(x_test, y_test))
svm_predictions = svm_pred[:81]

# PLOT IMAGES AND PREDICTIONS
plot_images(mlp_predictions, knn_predictions, dtc_predictions, gnb_predictions, svm_predictions)

'''
# Create a large window for easy viewing
plt.rcParams["figure.figsize"] = [16, 9]

# Set labels
plt.xlabel("INSTANCE")
plt.ylabel("VALUE")

# Create lines
print(y_test[:81])
plt.plot(mlp_pred[:81], color="Blue")
plt.plot(y_test[:81], color="Green")
# plt.plot(predicted, color="Red")

# Create dots
label = "MLP: " + "%.4f" % round(mlp_accuracy, 4)
plt.plot(mlp_pred[:81], "ro", color="Blue", label="MLP classifier")
plt.plot(y_test[:81], "ro", color="Green", label="Actual values")
# plt.plot(predicted, "ro", color="Red", label="Predicted")

# Prepare and show
plt.legend()
plt.show()
'''


