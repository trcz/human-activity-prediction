import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

#Function importing data from .txt file
def gatherData(directory):
    with open(directory, "rb") as myFile:
        data_table = pd.read_csv(myFile, sep=" ", header=None)
    return data_table.values

#Data importing

train_features = gatherData("train/X_train.txt")
train_labels = gatherData("train/Y_train.txt")
test_features = gatherData("test/X_test.txt")
test_labels = gatherData("test/Y_test.txt")
print("Data loaded")

#Model training

model = GaussianNB()
model.fit(train_features, train_labels.ravel())
print("Trained model 1")
model2 = LinearSVC(multi_class="ovr") #I'm using One vs Rest method for multi-class classification
model2.fit(train_features, train_labels.ravel())
print("Trained model 2")

#Making predictions

predictions = model.predict(test_features)
confusion = confusion_matrix(test_labels, predictions)
counter = 0
print("Predicted for model 1")

predictions2 = model2.predict(test_features)
confusion2 = confusion_matrix(test_labels, predictions2)
counter2 = 0
print("Predicted for model 2")

#Checking accuracy

for i in range(np.size(confusion, 0)):
    counter += confusion[i][i]
    counter2 += confusion2[i][i]
print("{0:.2f} percent of test values predicted well with GaussianNB".format(float(counter/len(predictions))*100))
print("{0:.2f} percent of test values predicted well with LinearSVC".format(float(counter2/len(predictions))*100))
