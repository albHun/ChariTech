import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import os
import glob


# The variable to store input pixels
X = list()
y = list()

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
count = 0
imageCount = 0
for i in range(1, 6):
	with open("Input{}.txt".format(i)) as fh:
		image = list()
		for line in fh:
			for j in line.split():
				if count == 262144:
					count = 0
					X.append(image)
					image = list()
					y.append(i)
					imageCount += 1
					if imageCount == 50:
						break 
				image.append(int(j))
				count += 1
				print(count)
		X.append(image)
		y.append(i)
				
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

y_vetorized = []
for out in y:


pca = PCA(n_component = 500, svd_solver = 'randomized',
	whiten = True).fit(X)
X_reduced = pca.transform(X)

clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5,
	hidden_layer_sizes = (10, 5), random_state = 1,
	activation = 'relu')
clf.fit(X_reduced, y)