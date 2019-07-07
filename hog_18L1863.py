from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import Counter

# Load the dataset
train = pd.read_csv('D:\Projects\MLProjects\mnist_train.csv')
test = pd.read_csv('D:\Projects\MLProjects\mnist_test.csv')

#Dividing the dataset
y_train = train.iloc[:,0].values
X_train = train.iloc[:,1:].values

y_test = test.iloc[:,0].values
X_test = test.iloc[:,1:].values

# Extraction of train hog features
list_hog_fd = []
for feature in X_train:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

print ("Count of digits in dataset", Counter(y_train))

# Extraction of test hog features
list_hog_fd = []
for feature in X_test:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features_test = np.array(list_hog_fd, 'float64')

print ("Count of digits in dataset", Counter(y_test))

# Create an linear SVM object
clf = LinearSVC()

# Perform the training
clf.fit(hog_features, y_train)

# Predict on Test dataset
y_pred = clf.predict(np.array(hog_features_test))

# Accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
print(score)

