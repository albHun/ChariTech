import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import os
import glob
import time
import operator
import random

X = list()
y = list()

imageCount = 0
sampleSize = 798


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
print(len(X))

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
print(len(X))

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
print(len(X))

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
print(len(X))

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
print(len(X))

time1 = time.time()
print(time1- time0)


def combineAndShuffle(X, y):
	combined = [(X[i], y[i]) for i in range(0, len(X))]
	random.shuffle(combined)

	X = [combined[i][0] for i in range(0, len(combined))]
	y = [combined[i][1] for i in range(0, len(combined))]
	return X, y


print("Machine Learning Begins")


X, y = combineAndShuffle(X, y)


for n_c in [300, 600, 1200, 2400]:
	time0 = time.time()
	pca = PCA(n_components = n_c, svd_solver = 'randomized',
		whiten = True).fit(X)
	X_reduced = pca.transform(X)
	X_reduced = MinMaxScaler().fit_transform(X_reduced)

	# Seperating training, cross validation and test set
	X_training = X_reduced[:int(len(X_reduced) * 0.8)]
	y_training = y[:int(len(X_reduced) * 0.8)]
	X_cv = X_reduced[int(len(X_reduced) * 0.8):]
	y_cv = y[int(len(y) * 0.8):]
	# print(X_cv)
	# print(y_cv)
	#X_test = X_reduced[int(len(X_reduced) * 0.8):]
	#y_test = y[int(len(y) * 0.8):]

	print(time.time() - time0)
	time0 = time.time()


	hidden_layer_size_selections = [(220, 110)]
	for hidden_layer_sizes in hidden_layer_size_selections:
		clf = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes, verbose = True, random_state=1, tol = 1e-7, max_iter = 200,
		                    solver= 'adam', learning_rate= 'constant', momentum= 0, alpha = 0.1)
		clf.fit(X_training, y_training)
		print("The hidden layer sizes are", hidden_layer_sizes)
		print(clf.score(X_training, y_training))
		print(clf.score(X_cv, y_cv))

	time1 = time.time()
	print(time1 - time0)
