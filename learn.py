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
count = 1
imageCount = 0
flag = False
time0 = time.time()
with open("A.txt") as fh:
	image = list()
	for line in fh:
		flag = False
		for j in line.split():
			if count == 262144:
				count = 0
				X.append(image)
				image = list()
				imageCount += 1
				print(imageCount)
				if imageCount == 50:
					flag = True
					break 
			image.append(int(j))
			count += 1
			#print(count)
		if flag:
			break
	X.append(image)
	y.append(1)
print(time.time() - time0)	
print(len(X))

count = 1
imageCount = 0
flag = False
time0 = time.time()
with open("L.txt") as fh:
	image = list()
	for line in fh:
		flag = False
		for j in line.split():
			if count == 262144:
				count = 0
				X.append(image)
				image = list()
				imageCount += 1
				print(imageCount)
				if imageCount == 50:
					flag = True
					break 
			image.append(int(j))
			count += 1
			#print(count)
		if flag:
			break
	X.append(image)
	y.append(0)
print(time.time() - time0)
print(len(X))
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
np_X = np.array(X)
np_Y = np.array(y)

pca = PCA(n_components = 500, svd_solver = 'randomized',
	whiten = True).fit(X)
X_reduced = pca.transform(X)

clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5,
	hidden_layer_sizes = (10, 5), random_state = 1,
	activation = 'relu')
clf.fit(X_reduced, y)






# Test
clf.predict(X_reduced)








