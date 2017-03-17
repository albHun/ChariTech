import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

fh = open("Input.txt")
#
#
#
#

# Some sort of input processing

# Get input X for each image
# Get input y for each image(class of fingerprint)
#
#
#


pca = PCA(n_component = 500, svd_solver = 'randomized',
	whiten = True).fit(X)
X_reduced = pca.transform(X)

clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5,
	hidden_layer_sizes = (10, 5), random_state = 1,
	activation = 'relu')
clf.fit(X_reduced, y)



