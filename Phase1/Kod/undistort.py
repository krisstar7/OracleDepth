import numpy as np 
import cv2 as cv
import glob

images = glob.glob('calibdata/*.jpg')

mtx = np.load('intrinsic.npy')
dist = np.load('distCoeff.npy')

img = cv.imread(images[0])

h,  w = img.shape[:2]
alpha = 1
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), alpha)

dst = cv.undistort(img, mtx, dist, None, newcameramtx)

cv.namedWindow("dst", cv.WINDOW_NORMAL) 
dst = cv.resize(dst, (1920, 1080))

cv.imshow('dst', dst)
cv.waitKey()

