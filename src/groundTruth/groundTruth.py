"""
    File name: groundTruth.py
    Author: Animesh Garg
    Date created: 3/16/2016
    Date last modified: 6/29/2016
    Python Version: 2.7
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage import data
import cv2

from matplotlib import pyplot as plt


# if os.path.isfile('sample.jpg'): im = array(Image.open('sample.jpg'))
img = cv2.imread('src/groundTruth/test.png',0)

ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
# ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,thresh5 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# blur = cv2.GaussianBlur(thresh5,(10,5),0)

thresh6 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

# titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','OTSU','Gauss_adap']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5, thresh6]

for i in xrange(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#     plt.subplot(2,3,i+1),plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()

kernel = np.ones((9,9),np.uint8)
opening = cv2.morphologyEx(thresh5, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

titles = ['Original Image','OTSU','Gauss_adap','Open', 'open+close']
images = [img, thresh5, thresh6, opening, closing]


for i in xrange(5):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

boundary = np.vstack(contours[1]).squeeze()
np.savetxt('boundary.out', boundary)

cv2.drawContours(img,contours,-1,(255,255,255),1)
cv2.imshow("Contour",img)
# cv2.waitKey(0)