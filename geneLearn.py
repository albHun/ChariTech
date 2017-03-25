# General learning program
# Setting things up and perform PCA on data


import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from dataInput import readFingerprintData
from learnSVM import svmLearning


time0 = time.time()

X, y = readFingerprintData()
# Setting PCA parameters
n_c = [200]
for n_feature in n_c:
	time1 = time.time()
	#time1 = time.time()
	#print("The reduced number of features is ", n_feature)
	#pca = PCA(n_components = n_feature, svd_solver = 'randomized',
	#		whiten = True).fit(X)
	#X_reduced = pca.transform(X)
	#X_reduced = MinMaxScaler().fit_transform(X_reduced)

	# Separating training and cross validation sets
	X_training = X[:int(len(X) * 0.9)]
	y_training = y[:int(len(X) * 0.9)]
	X_cv = X[int(len(X) * 0.9):]
	y_cv = y[int(len(y) * 0.9):]

	# Performing svm and see the results
	svmLearning(X_training, y_training, X_cv, y_cv)
	print(time.time() - time1)
print(time.time() - time0)