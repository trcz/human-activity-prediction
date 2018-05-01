import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


def gatherData(directory):
    with open(directory, "rb") as myFile:
        data_table = pd.read_csv(myFile, sep=" ", header=None)
    return data_table.values


train_features = gatherData("train/X_train.txt")
train_labels = gatherData("train/Y_train.txt")
test_features = gatherData("test/X_test.txt")
test_labels = gatherData("test/Y_test.txt")

model = GaussianNB()
model.fit(train_features, train_labels.ravel())

predictions = model.predict(test_features)
confusion = confusion_matrix(test_labels, predictions)
counter = 0
for i in range(np.size(confusion, 0)):
    counter += confusion[i][i]
print("{0} percent of test values predicted well".format(float(counter/len(predictions))*100))