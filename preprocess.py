# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 22:48:07 2019

@author: William
"""

import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('wall.jpg')[150:1000,:1800,:]
cv2.namedWindow('original')
cv2.imshow('original', img)

rows,cols,ch = img.shape

#pts1 = np.float32([[56,65], [368,52], [28,387], [389,390]])
#pts2 = np.float32([[0,0], [300,0], [0,300], [300,300]])
#
#M = cv2.getPerspectiveTransform(pts1, pts2)
#
#dst = cv2.warpPerspective(img, M, (300,300))
#
#plt.subplot(121),plt.imshow(img),plt.title('Input')
#plt.subplot(122),plt.imshow(dst),plt.title('Output')
#plt.show()
#%%
img_part = img[:, :, :]
blurred = cv2.GaussianBlur(img_part, (3, 3), 0)
#cv2.namedWindow('blurred')
#cv2.imshow('blurred', blurred)

hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
#white_mask = cv2.inRange(hsv, (78,43,46), (99,255,255))
white_mask = cv2.inRange(hsv, (0,0,200), (180,40,255))
white_masks = np.repeat(white_mask[:, :, np.newaxis], 3, axis=2)
img_masked1 = img_part.copy()
#img_masked[white_masks == 0] = 0

contours, hierarchy = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
rec = []
area = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    #print('x={}, y={}, width={}, height={}'.format(x, y, w, h))
    #cv2.rectangle(img_masked, (x,y), (x+w, y+h), (0,255,0), 2)
    rec.append([x,y,w,h])
    area.append(w*h)
Max=0
for i in range(1,len(area)):
    if area[i]>area[Max]: Max = i
#cv2.rectangle(img_masked, (x,y), (x+w, y+h), (0,255,0), 2)
cv2.rectangle(img_masked1, (rec[Max][0],rec[Max][1]), (rec[Max][0]+rec[Max][2], rec[Max][1]+rec[Max][3]), (0,255,0), 2)
print('x={}, y={}, width={}, height={}'.format(rec[Max][0],rec[Max][1],rec[Max][2],rec[Max][3]))
#%%
white_mask = cv2.inRange(hsv, (78,43,46), (99,255,255))
white_masks = np.repeat(white_mask[:, :, np.newaxis], 3, axis=2)
img_masked2 = img_part.copy()
#img_masked[white_masks == 0] = 0

contours, hierarchy = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
rec = []
area = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    #print('x={}, y={}, width={}, height={}'.format(x, y, w, h))
    #cv2.rectangle(img_masked, (x,y), (x+w, y+h), (0,255,0), 2)
    rec.append([x,y,w,h])
    area.append(w*h)
Max=0
for i in range(1,len(area)):
    if area[i]>area[Max]: Max = i
#cv2.rectangle(img_masked, (x,y), (x+w, y+h), (0,255,0), 2)
cv2.rectangle(img_masked2, (rec[Max][0],rec[Max][1]), (rec[Max][0]+rec[Max][2], rec[Max][1]+rec[Max][3]), (0,255,0), 2)
print('x={}, y={}, width={}, height={}'.format(rec[Max][0],rec[Max][1],rec[Max][2],rec[Max][3]))
#%%
cv2.namedWindow('masked')
cv2.imshow('masked', img_masked1|img_masked2)
cv2.waitKey()
#%%