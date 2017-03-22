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
sampleSize = 1990




time0 = time.time()
with open(r"fingerprintClassification\thresholded_text\concatenated_line_breaked\A.txt") as fh:
	for line in fh.readlines():
		imageCount += 1
		if imageCount >= sampleSize:
			break
		ints = [int(x) for x in line if x != ' ' and x != '\n']
		X.append(ints)
		y.append(1)



imageCount = 0
time0 = time.time()
with open(r"fingerprintClassification\thresholded_text\concatenated_line_breaked\L.txt") as fh:
	for line in fh.readlines():
		imageCount += 1
		if imageCount >= sampleSize:
			break
		ints = [int(x) for x in line if x != ' ' and x != '\n']
		X.append(ints)
		y.append(0)


imageCount = 0
time0 = time.time()
with open(r"fingerprintClassification\thresholded_text\concatenated_line_breaked\R.txt") as fh:
	for line in fh.readlines():
		imageCount += 1
		if imageCount >= sampleSize:
			break
		ints = [int(x) for x in line if x != ' '  and x != '\n']
		X.append(ints)
		y.append(0)

imageCount = 0
time0 = time.time()
with open(r"fingerprintClassification\thresholded_text\concatenated_line_breaked\T.txt") as fh:
	for line in fh.readlines():
		imageCount += 1
		if imageCount >= sampleSize:
			break
		ints = [int(x) for x in line if x != ' ' and x != '\n']
		X.append(ints)
		y.append(0)


imageCount = 0
time0 = time.time()
with open(r"fingerprintClassification\thresholded_text\concatenated_line_breaked\W.txt") as fh:
	for line in fh:
		imageCount += 1
		if imageCount >= sampleSize:
			break
		ints = [int(x) for x in line if x != ' '  and x != '\n']
		X.append(ints)
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

print("Machine Learning Begins")

pca = PCA(n_components = 192, svd_solver = 'randomized',
	whiten = True).fit(X)
X_reduced = pca.transform(X)


clf = MLPClassifier(solver = 'adam', alpha = 1e-5, early_stopping = True,
	hidden_layer_sizes = (100, 20, 4), random_state = 1,
	activation = 'relu')
clf.fit(X_reduced, y)

# Test
sum = 0
for i in range(0, 50):
	sum += clf.predict(X_reduced)[i]
	#print(clf.predict_proba(X_reduced)[i])
#print(sum/50)

sum = 0
for i in range(50, 100):
	sum += clf.predict(X_reduced)[i]
	#print(clf.predict_proba(X_reduced)[i])
#print(1- sum/50)

#print(time.time() - time1)
scores = cross_val_score(clf, X_reduced, y)
print(scores.mean())

