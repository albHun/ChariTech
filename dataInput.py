import time
import random

def readFingerprintData():
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

	X, y = combineAndShuffle(X, y)
	return X, y