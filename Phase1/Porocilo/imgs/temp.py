import numpy as np
import cv2 as cv
import glob


width = 5
height = 7
square_len = 27.2/1000
marker_len = 14/1000
dict = 7

#board
aruco_dict = cv.aruco.getPredefinedDictionary(dict)
board_size = (width, height)
board = cv.aruco.CharucoBoard(board_size, square_len, marker_len, aruco_dict)

#refine params
refparams = cv.aruco.RefineParameters(minRepDistance = 10.,
									  errorCorrectionRate = 3.,
									  checkAllOrders=True)

charucoparms = cv.aruco.CharucoParameters()
charucoparms.tryRefineMarkers = True



charuco_detector = cv.aruco.CharucoDetector(board, refineParams=refparams, charucoParams=charucoparms)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*4,3), np.float32)
objp[:,:2] = np.mgrid[0:4,0:6].T.reshape(-1,2)
# objp = list(objp)
# Arrays to store object points and image points from all the images.
objpoints = []
imgpoints = []


img = cv.imread('charuco.png')
h,  w = img.shape[:2]
 
charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(img)

imgpoints.append(charuco_corners)

objtemp = []
for id in charuco_ids:
	objtemp.append(objp[int(id)])
	objpoints.append(np.array(objtemp))
 
if not (charuco_ids is None) and len(charuco_ids) > 0:
	cv.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
	cv.aruco.drawDetectedMarkers(img, marker_corners, marker_ids, (0,0,255))
	cv.imshow(str(len(charuco_ids)), img)
	cv.waitKey(2000)
 
cv.imwrite('charucoDetect.png', img)