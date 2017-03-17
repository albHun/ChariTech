from PIL import Image
import numpy as np
from time import time
import os

edgeL = 64
edgeH = 448

compressed_width = (edgeH - edgeL) // 2


def clip(n, minimum=0, maximum=255):
    return max(min(maximum, n), minimum)


def threshold_128_text(in_path="trainingSet/A/f0005_03.png", out_path="thresholded_text/A/f0005_03.txt"):
    imori = Image.open(in_path)
    pxori = imori.load()  # pixel object to access pixels
    immod = Image.new('L', imori.size)
    pxmod = immod.load()
    # imcom = Image.new('L', (compressed_width, compressed_width))
    # pxcom = imcom.load()

    step = 16
    sqStep = step * step
    ignBd = 32
    ev = 0
    bound = 0
    Min = 0
    Max = 0

    for i in range(edgeL, edgeH, step):
        for j in range(edgeL, edgeH, step):
            Min = 255
            Max = 0
            ev = 0
            for di in range(step):
                for dj in range(step):
                    ev += pxori[i + di, j + dj]
                    Min = min(Min, pxori[i + di, j + dj])
                    Max = max(Max, pxori[i + di, j + dj])
            bound = ev // sqStep
            if(Max - Min < ignBd):
                for di in range(step):
                    for dj in range(step):
                        pxmod[i + di, j + dj] = 255
            else:
                for di in range(step):
                    for dj in range(step):
                        if(pxori[i + di, j + dj] > bound):
                            pxmod[i + di, j + dj] = 255

    step = 2
    bound = step * step / 2

    outstring = ''
    for j in range(edgeL, edgeH, step):
        for i in range(edgeL, edgeH, step):
            ev = 0
            for di in range(step):
                for dj in range(step):
                    if(pxmod[i + di, j + dj] == 255):
                        ev += 1
            if(ev > bound):
                # pxcom[(i - edgeL) / step, (j - edgeL) / step] = 255
                outstring += '1 '
            else:
                outstring += '0 '
        outstring += '\n'

    with open(out_path, 'w') as outfile:
        outfile.write(outstring)
    # imcom.save(out_path)


def threshold_128(in_path="trainingSet/A/f0005_03.png", out_path="thresholded/A/f0005_03.png"):
    imori = Image.open(in_path)
    pxori = imori.load()  # pixel object to access pixels
    immod = Image.new('L', imori.size)
    pxmod = immod.load()
    imcom = Image.new('L', (compressed_width, compressed_width))
    pxcom = imcom.load()

    step = 16
    sqStep = step * step
    ignBd = 32
    ev = 0
    bound = 0
    Min = 0
    Max = 0

    for i in range(edgeL, edgeH, step):
        for j in range(edgeL, edgeH, step):
            Min = 255
            Max = 0
            ev = 0
            for di in range(step):
                for dj in range(step):
                    ev += pxori[i + di, j + dj]
                    Min = min(Min, pxori[i + di, j + dj])
                    Max = max(Max, pxori[i + di, j + dj])
            bound = ev // sqStep
            if(Max - Min < ignBd):
                for di in range(step):
                    for dj in range(step):
                        pxmod[i + di, j + dj] = 255
            else:
                for di in range(step):
                    for dj in range(step):
                        if(pxori[i + di, j + dj] > bound):
                            pxmod[i + di, j + dj] = 255

    step = 2
    bound = step * step / 2

    for j in range(edgeL, edgeH, step):
        for i in range(edgeL, edgeH, step):
            ev = 0
            for di in range(step):
                for dj in range(step):
                    if(pxmod[i + di, j + dj] == 255):
                        ev += 1
            if(ev > bound):
                pxcom[(i - edgeL) / step, (j - edgeL) / step] = 255
    imcom.save(out_path)


# Test
# t = time()
# threshold_128()
# print('Done in', time() - t, 's')
###############

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
    for i, f in enumerate(current_files):
        if i > 10:
            break
        print(i)
        threshold_128(in_path='trainingSet/{}/{}'.format(current, f),
                      out_path="thresholded/{}/{}.png".format(current, f[:-4]))
    print("{} done.".format(current))
