from PIL import Image
# import numpy as np
from time import time
import os


def threshold(in_path="trainingSet/A/f0005_03.png", out_path="thresholded/A/f0005_03.png"):
    """Convert the image in the in_path tout_patho 1-bit grayscale image using threshold value t"""
    im = Image.open(in_path)
    px = im.load()  # pixel object to access pixels

    minimum = 255
    maximum = 0
      
    for x in range(170, 340):
        minimum = min(minimum, px[x, 255])
        maximum = max(maximum, px[x, 255])
    t = (minimum + maximum) // 2

    bw = im.point(lambda x: 0 if x < t else 255, '1')
    bw.save(out_path)

def thresholdR(in_path="trainingSet/A/f0005_03.png", out_path="thresholded/A/f0005_03.png"):
    im = Image.open(in_path)
    px = im.load()  # pixel object to access pixels
    bw = Image.new('L', im.size)
    bwx = bw.load()
    step = 8
    yo = 0
    minimum = 255
    maximum = 0
    bound = 0
    while yo + step <= 512:
        minimum = 255
        maximum = 0
        
        for x in range(170, 340) :
            minimum = min(minimum, px[x, yo+step//2])
            maximum = max(maximum, px[x, yo+step//2])
        bound = (minimum + maximum) // 2
        for y in range(yo,yo + step):
            for x in range(512):
                if(px[x,y]< bound):
                    bwx[x,y] = 0
                else:
                    bwx[x,y] = 255
        yo += step
    bw.save(out_path);
           
def thresholdT(in_path="trainingSet/A/f0005_03.png", out_path="thresholded/A/f0005_03.png"):
    imori = Image.open(in_path)
    pxori = imori.load()  # pixel object to access pixels
    immod = Image.new('L', imori.size)
    pxmod = immod.load()
    
    step = 8
    sqStep = step * step
    ev = 0
    bound = 0
    
    for i in range(0, 512, step):
        for j in range(0, 512, step):
            ev = 0
            for di in range(step):
                for dj in range(step):
                    ev += pxori[i+di,j+dj]
            bound = ev//sqStep
            print(bound)
            for di in range(step):
                for dj in range(step):
                    if(pxori[i+di,j+dj]<bound):
                        pxmod[i+di,j+dj] = 255
    immod.save(out_path);

# t = time()
# threshold(128)
# print('Done in', time() - t, 's')
filesA = [f for f in os.listdir('trainingSet/A') if f[-4:] == '.png']
filesL = [f for f in os.listdir('trainingSet/L') if f[-4:] == '.png']
filesR = [f for f in os.listdir('trainingSet/R') if f[-4:] == '.png']
filesT = [f for f in os.listdir('trainingSet/AT') if f[-4:] == '.png']
filesW = [f for f in os.listdir('trainingSet/W') if f[-4:] == '.png']

for f in filesT:
    thresholdT(in_path='trainingSet/AT/{}'.format(f), out_path="thresholded/ATR/{}".format(f))
print("T done.")

for f in filesT:
    threshold(in_path='trainingSet/AT/{}'.format(f), out_path="thresholded/AT/{}".format(f))
print("T done.")
