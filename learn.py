import numpy as np
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
count = 0
imageCount = 0
totalPixels = 262144
sampleSize = 500





flag = False
time0 = time.time()
with open("F:\MyProjects\ChariTechData\A.txt") as fh:
	image = list()
	for line in fh:
		flag = False
		for j in line.split():
			if count == totalPixels:
				count = 0
				X.append(image)
				y.append(1)
				image = list()
				imageCount += 1
				if imageCount == sampleSize:
					flag = True
					break 
			image.append(int(j))
			count += 1
			#print(count)
		if flag:
			break
	if len(image) > 0:
		X.append(image)
		y.append(1)
print(time.time() - time0)	
print(len(X))
print(len(y))

count = 0
imageCount = 0
flag = False
time0 = time.time()
with open("F:\MyProjects\ChariTechData\A.txt") as fh:
	image = list()
	for line in fh:
		flag = False
		for j in line.split():
			if count == totalPixels:
				count = 0
				X.append(image)
				y.append(0)
				image = list()
				imageCount += 1
				if imageCount == sampleSize:
					flag = True
					break 
			image.append(int(j))
			count += 1
			#print(count)
		if flag:
			break
	if len(image) > 0:
		X.append(image)
		y.append(0)

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


pca = PCA(n_components = 2000, svd_solver = 'randomized',
	whiten = True).fit(X)
X_reduced = pca.transform(X)
print(len(X_reduced))

clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5,
	hidden_layer_sizes = (166, 50, 16), random_state = 1,
	activation = 'relu')
clf.fit(X_reduced, y)

# Test
sum = 0
for i in range(0, 50):
	sum += clf.predict(X_reduced)[i]
	print(clf.predict_proba(X_reduced)[i])
print(sum/50)

sum = 0
for i in range(50, 100):
	sum += clf.predict(X_reduced)[i]
	print(clf.predict_proba(X_reduced)[i])
print(1- sum/50)

print(time.time() - time1)


