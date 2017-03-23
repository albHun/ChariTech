f = open(r'fingerprintClassification\thresholded_text\concatenated_single_lines\A.txt')

for line in f:
	for char in line:
		print(char)