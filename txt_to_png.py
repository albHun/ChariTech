from PIL import Image


im = Image.new('L', (192, 192))
px = im.load()
with open("thresholded_text/A/f0005_03.txt", 'r') as infile:
    lines = infile.readlines()
    for i, line in enumerate(lines):
        for j, x in enumerate(line):
            pass