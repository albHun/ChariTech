import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import os
import glob
import time

# The variable to store input pixels


# Read all text files in the directory
"""
file_path = 'thresholded_text/A'
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
sampleSize = 50

# f = open("fingerprintClassification/thresholded_text/concatenated_single_lines/A.txt",'r')

# for line in f.readlines():

#   for char in line:
#       print(char)


time0 = time.time()
with open("fingerprintClassification/thresholded_text/concatenated_single_lines/A.txt") as fh:
    lines = fh.readlines()
    for line in lines:
        imageCount += 1
        # if imageCount >= sampleSize:
        #     break
        ints = [int(x) for x in line[:-1]]
        X.append(ints)
        y.append(1)
print(time.time() - time0)
print(len(X))
print(len(y))


imageCount = 0
time0 = time.time()
with open("fingerprintClassification/thresholded_text/concatenated_single_lines/L.txt") as fh:
    for line in fh.readlines():
        imageCount += 1
        # if imageCount >= sampleSize:
        #     break
        ints = [int(x) for x in line[:-1]]
        X.append(ints)
        y.append(0)
print(time.time() - time0)
print(len(X))
print(len(y))


imageCount = 0
time0 = time.time()
with open("fingerprintClassification/thresholded_text/concatenated_single_lines/R.txt") as fh:
    for line in fh.readlines():
        imageCount += 1
        # if imageCount >= sampleSize:
        #     break
        ints = [int(x) for x in line[:-1]]
        X.append(ints)
        y.append(0)
print(time.time() - time0)
print(len(X))
print(len(y))

imageCount = 0
time0 = time.time()
with open("fingerprintClassification/thresholded_text/concatenated_single_lines/T.txt") as fh:
    for line in fh.readlines():
        imageCount += 1
        # if imageCount >= sampleSize:
        #     break
        ints = [int(x) for x in line[:-1]]
        X.append(ints)
        y.append(0)
print(time.time() - time0)
print(len(X))
print(len(y))


imageCount = 0
time0 = time.time()
with open("fingerprintClassification/thresholded_text/concatenated_single_lines/W.txt") as fh:
    for line in fh:
        imageCount += 1
        # if imageCount >= sampleSize:
        #     break
        ints = [int(x) for x in line[:-1]]
        X.append(ints)
        y.append(0)
print(time.time() - time0)
print(len(X))
print(len(y))

time1 = time.time()
print(time1 - time0)
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

pca = PCA(n_components=50, svd_solver='randomized',
          whiten=True).fit(X)
X_reduced = pca.transform(X)
print(len(X_reduced))

clf = MLPClassifier(solver='adam', alpha=1e-5, early_stopping=True,
                    hidden_layer_sizes=(30, 16), random_state=1,
                    activation='relu')
clf.fit(X_reduced, y)

# Test
sum = 0
for i in range(0, 50):
    sum += clf.predict(X_reduced)[i]
    print(clf.predict_proba(X_reduced)[i])
print(sum / 50)

sum = 0
for i in range(50, 100):
    sum += clf.predict(X_reduced)[i]
    print(clf.predict_proba(X_reduced)[i])
print(1 - sum / 50)

print(time.time() - time1)
