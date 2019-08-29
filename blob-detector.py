def calcSigma(no_s,k,s):
    sigma = []
    for i in range(no_s):
        sigma1 = s*(k**i)
        sigma.append(sigma1)

    h = [round(i*6) for i in sigma]
    for i in range(len(h)):
        if h[i]%2 == 0:
            h[i] = h[i]+1
    return sigma,h


def calcLoG(k,sigma,no_s,h):
    LoG = []
    for i in range(no_s):
        LoG1 = np.zeros((h[i],h[i]))
        r1 = int(-(math.floor(h[i]/2)))
        r2 = int(math.floor(h[i]/2))
        for x in range(r1,r2+1):
            for y in range(r1,r2+1):
                LoG1[x+r2,y+r2] = (-1/(math.pi*(sigma[i]**2)))*(1-(((x**2)+(y**2))/(2*(sigma[i]**2))))*np.exp(-((x**2)+(y**2))/(2*(sigma[i]**2)))
        LoG.append(LoG1)
    return LoG


def convPadded(image, Filter,no_s,h):
    start = time.time()
    row = image.shape[0]
    col = image.shape[1]
    convout = []
    for i in range(no_s):
        Filter[i] = np.flipud(np.fliplr(Filter[i]))  
        out = np.zeros_like(image)  
        r = int(math.floor(h[i]/2))+1
        padded = np.zeros((row + h[i]+1, col + h[i]+1))   
        padded[r:-r, r:-r] = image
        for x in range(row): 
            for y in range(col):
                out[x,y]=(Filter[i]*padded[x:x+h[i],y:y+h[i]]).sum()  
        convout.append(out)
    end = time.time()
    print('Time taken for convolution : ', end-start)
    return convout 


def maximaDetection(img,convout,sigma,no_s,threshold):
    start1 = time.time()
    row = img.shape[0]
    col = img.shape[1]
    blob_loc = []
    no_minimas = 0
    for n in range(no_s):
        blob = []
        rad = int(math.sqrt(2)*sigma[n])
        for x in range(row):
            for y in range(col):
                temp = 'minima'
                for o in range(-1,2):
                    for i in range(-1,2):
                        for j in range(-1,2):
                            if x+i >= 0 and y+j >= 0 and x+i < row and y+j < col and n+o >= 0 and n+o < no_s:
                                if convout[n][x,y] > threshold:
                                    if convout[n][x,y] < convout[n+o][x+i,y+j]: temp = 'not minima'   
                                else: temp = 'not minima'
                if temp == 'minima':
                    if x-rad > 0 and y-rad > 0 and x+rad < row-1 and y+rad < col-1:
                        blob.append([x,y])
                        cv2.circle(img, (y, x), rad, (0,0,255), 1)
        print('No of minimas in scale space', n,': ',len(blob))
        no_minimas = no_minimas + len(blob)
        blob_loc.append(blob)
    end1 = time.time()
    print('Time taken for maxima reduction : ', end1-start1)
    print('Total no. of minimas : ', no_minimas)
    return img


import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import time

START = time.time()
img = cv2.imread('Images/sunflowers.jpg', 0)
img1 = cv2.imread('Images/sunflowers.jpg')
img = img/255
print('Image shape : ',img.shape)
plt.imshow(img, cmap = 'gray')
plt.show()

no_s = 15
k = 1.24
s = 1/math.sqrt(2.5)
threshold = 0.02

sigma,h = calcSigma(no_s,k,s)
LoG = calcLoG(k,sigma,no_s,h)\
convout = convPadded(img,LoG,no_s,h)
img1 = maximaDetection(img1,convout,sigma,no_s,threshold)

plt.imshow(img1)
plt.show()
cv2.imwrite('Blob Images/sunflower-blobs.png',img1)
print('Time taken for the complete program = ', time.time() - START, 'sec')

