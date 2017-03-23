import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import os
import glob
import time

# The variable to store input pixels


# Read all text files in the directory
"""
file_path = 'thresholded_text\A'
count = 0
for filename in glob.glob(os.path.join(file_path, '*.txt')):
	with open(filename) as fh:
		image = list()
		for line in fh:
			count += 1
			print(count)
			for j in line.split():
				image.append(int(j))
		X.append(image)
print("DONE")
"""


X = list()
y = list()

# One vs all: if this image is in class A or L
# Using 50 images

imageCount = 0
sampleSize = 200




time0 = time.time()
with open("fingerprintClassification/thresholded_text/concatenated_single_lines\A.txt") as fh:
	for line in fh.readlines():
		imageCount += 1
		if imageCount >= sampleSize:
			break
		ints = [int(x) for x in line if x != ' ' and x != '\n']
		X.append(ints)
		y.append(1)
print("Data collected")



imageCount = 0
with open("fingerprintClassification/thresholded_text/concatenated_single_lines\L.txt") as fh:
	for line in fh.readlines():
		imageCount += 1
		if imageCount >= sampleSize:
			break
		ints = [int(x) for x in line if x != ' ' and x != '\n']
		X.append(ints)
		y.append(2)
print("Data collected")

imageCount = 0
with open("fingerprintClassification/thresholded_text/concatenated_single_lines\R.txt") as fh:
	for line in fh.readlines():
		imageCount += 1
		if imageCount >= sampleSize:
			break
		ints = [int(x) for x in line if x != ' '  and x != '\n']
		X.append(ints)
		y.append(3)
print("Data collected")

imageCount = 0
with open("fingerprintClassification/thresholded_text/concatenated_single_lines\T.txt") as fh:
	for line in fh.readlines():
		imageCount += 1
		if imageCount >= sampleSize:
			break
		ints = [int(x) for x in line if x != ' ' and x != '\n']
		X.append(ints)
		y.append(4)
print("Data collected")

imageCount = 0
with open("fingerprintClassification/thresholded_text/concatenated_single_lines\W.txt") as fh:
	for line in fh:
		imageCount += 1
		if imageCount >= sampleSize:
			break
		ints = [int(x) for x in line if x != ' '  and x != '\n']
		X.append(ints)
		y.append(5)
print("Data collected")

time1 = time.time()
print(time1- time0)
#
#
#
#

# Some sort of input processing

# Get input X for each image

# Get output y for each image(class of fingerprint)
#
#
#

def oneVsAllClassifierPrepareY(classIndex, y):
	y_transformed = np.array(len(y))
	for i in range(0, len(y)):
		if y[i] != classIndex:
			y_transformed[i] = 0
		else:
			y_transformed[i] = 1
	return y_transformed

def oneVsAllClassifier(classIndex, X, y):
	clf = MLPClassifier(solver = 'adam', alpha = 1e-5, early_stopping = True,
	hidden_layer_sizes = (100, 20, 4), random_state = 1,
	activation = 'relu')
	clf.fit(X, oneVsAllClassifierPrepareY(classIndex, y))
	return clf



print("Machine Learning Begins")



pca = PCA(n_components = 200, svd_solver = 'randomized',
	whiten = True).fit(X)
X_reduced = pca.transform(X)


# Seperating training, cross validation and test set
X_training = X_reduced[:len(X_reduced) * 0.6]
y_training = y[:len(X_reduced) * 0.6]
X_cv = X_reduced[len(X_reduced) * 0.6:len(X_reduced) * 0.8]
y_cv = y[len(y) * 0.6:len(y) * 0.8]
X_test = X_reduced[len(X_reduced) * 0.8:]
y_test = y[len(y) * 0.8:]


# Training 5 classifiers and select the one with highest probability
clf = oneVsAllClassifier(1, X_training, y_training)
prob1 = clf.predict_proba(X_cv)

clf = oneVsAllClassifier(2, X_training, y_training)
prob2 = clf.predict_proba(X_cv)

clf = oneVsAllClassifier(3, X_training, y_training)
prob3 = clf.predict_proba(X_cv)

clf = oneVsAllClassifier(4, X_training, y_training)
prob4 = clf.predict_proba(X_cv)

clf = oneVsAllClassifier(5, X_training, y_training)
prob5 = clf.predict_proba(X_cv)

selectionDictionary = {1:prob1, 2:prob2, 3:prob3, 4:prob4, 5:prob5}
for i in range(0, len(prob1)):
	probList = [prob1[i], prob2[i], prob3[i], prob4[i], prob5[i]]
