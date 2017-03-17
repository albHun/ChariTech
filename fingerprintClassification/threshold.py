from PIL import Image
import numpy as np
from time import time
import os


def clip(n, minimum=0, maximum=255):
    return max(min(maximum, n), minimum)


def threshold_text(in_path="trainingSet/A/f0005_03.png", out_path="thresholded_text/A/f0005_03.txt"):
    """Convert the image in the in_path tout_patho 1-bit grayscale image using threshold value t"""
    im = Image.open(in_path)
    px = im.load()  # pixel object to access pixels

    # edge detection
    # edged = np.zeros(im.shape)

    minimum = 255
    maximum = 0
    for x in range(170, 340):
        minimum = min(minimum, px[x, 255])
        maximum = max(maximum, px[x, 255])

    t = (minimum + maximum) // 2

    bw = im.point(lambda x: 0 if x < t else 255, '1')

    # load as numpy array
    arr = np.array(bw)
    # print(type(arr), arr)

    # save numpy array to text file
    np.savetxt(out_path, arr, fmt='%d')

    # bw.save(out_path)


def threshold(in_path="trainingSet/A/f0005_03.png", out_path="thresholded_text/A/f0005_03.png"):
    """Convert the image in the in_path tout_patho 1-bit grayscale image using threshold value t"""
    im = Image.open(in_path)
    px = im.load()  # pixel object to access pixels

    # edge detection
    # edged = np.zeros(im.shape)

    minimum = 255
    maximum = 0
    for x in range(170, 340):
        minimum = min(minimum, px[x, 255])
        maximum = max(maximum, px[x, 255])

    t = (minimum + maximum) // 2

    bw = im.point(lambda x: 0 if x < t else 255, '1')
    bw.save(out_path)


# t = time()
# threshold_text(out_path="thresholded_text/A/f0005_03.txt")
# print('Done in', time() - t, 's')

filesA = [f for f in os.listdir('trainingSet/A') if f[-4:] == '.png']
filesL = [f for f in os.listdir('trainingSet/L') if f[-4:] == '.png']
filesR = [f for f in os.listdir('trainingSet/R') if f[-4:] == '.png']
filesT = [f for f in os.listdir('trainingSet/T') if f[-4:] == '.png']
filesW = [f for f in os.listdir('trainingSet/W') if f[-4:] == '.png']

current = 'W'

if current == 'A':
    current_files = filesA
elif current == 'L':
    current_files = filesL
elif current == 'R':
    current_files = filesR
elif current == 'T':
    current_files = filesT
elif current == 'W':
    current_files = filesW


if __name__ == '__main__':
    for f in current_files:
        threshold_text(in_path='trainingSet/{}/{}'.format(current, f),
                       out_path="thresholded_text/{}/{}.txt".format(current, f[:-4]))
    print("{} done.".format(current))
