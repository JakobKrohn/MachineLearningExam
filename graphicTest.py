import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
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


def plot_images(mlp_score):

    d = x_test[0]
    d.shape = (28, 28)

    columns = 5
    rows = 5

    plt.rcParams["figure.figsize"] = [14, 6]
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

        figure.text((rows + 30), columns - 10, get_text("MLP", mlp_score[i]), color=get_color(label, mlp_score[i]))
        figure.text((rows + 30), (columns + 0), "KNN: x.xxx")
        figure.text((rows + 30), (columns + 10), "DTC: x.xxx")
        figure.text((rows + 30), (columns + 20), "GNB: x.xxx")
        figure.text((rows + 30), (columns + 30), "SVM: x.xxx")

    plt.show()


df = pd.read_csv("train.csv").as_matrix()

print(df.shape)

x = df[0:, 1:]
y = df[0:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y)

# MLP CLASSIFIER
print("\nTraining MLP classifier")
mlp = MLPClassifier(verbose=True, solver='adam', activation='relu', learning_rate='constant', hidden_layer_sizes=(783,))
# mlp.fit(x_train[:10], y_train[:10])
mlp.fit(x_train, y_train)
mlp_pred = mlp.predict(x_test)
mlp_accuracy = accuracy_score(y_test, mlp_pred)
print("Accuracy score: %f" % accuracy_score(y_test, mlp_pred))
print("Training set score: %f" % mlp.score(x_train, y_train))
print("Test set score: %f" % mlp.score(x_test, y_test))
mlp_predictions = mlp_pred[:26]


plot_images(mlp_predictions)

# Create a large window for easy viewing
plt.rcParams["figure.figsize"] = [16, 9]

# Set labels
plt.xlabel("INSTANCE")
plt.ylabel("VALUE")

# Create lines
print(y_test[1:26])
plt.plot(mlp_pred[1:26], color="Blue")
plt.plot(y_test[1:26], color="Green")
# plt.plot(predicted, color="Red")

# Create dots
label = "MLP: " + "%.4f" % round(mlp_accuracy, 4)
plt.plot(mlp_pred[1:26], "ro", color="Blue", label="MLP classifier")
plt.plot(y_test[1:26], "ro", color="Green", label="Actual values")
# plt.plot(predicted, "ro", color="Red", label="Predicted")

# Prepare and show
plt.legend()
plt.show()



