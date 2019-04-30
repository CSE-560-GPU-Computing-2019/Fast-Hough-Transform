import cv2
import time
import numpy as np

img = cv2.imread("input.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

begin = time.time()
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,1,param1=50,param2=20,minRadius=0,maxRadius=0)
end = time.time()
print("Time for hough circle", end-begin)
