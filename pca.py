import numpy as np
from sklearn.decomposition import PCA

fh = open("Input.txt")
#
#
#
#

# Some sort of input processing

# Get input X
#
#
#
#


pca = PCA(n_component = 500, svd_solver = 'randomized',
	whiten = True).fit(X)
X_reduced = pca.transform(X)
