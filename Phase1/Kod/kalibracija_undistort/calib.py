import numpy as np
import cv2 as cv
import glob

width = 7
height = 10
square_len = 27.2/1000
marker_len = 14/1000
dict = 12

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

# object points (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros(((height-1)*(width-1),3), np.float32)
objp[:,:2] = np.mgrid[0:(width-1),0:(height-1)].T.reshape(-1,2)

objpoints = []
imgpoints = []

images = glob.glob('calibdata/*.jpg')

for fname in images:

	img = cv.imread(fname)
 
	h,  w = img.shape[:2]

	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	# ret, gray = cv.threshold(gray, 127,255, cv.THRESH_BINARY) #zbolemuva error
 
	charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

	imgpoints.append(charuco_corners)
	
	objtemp = []
	for id in charuco_ids:
		objtemp.append(objp[int(id)])
  
	objpoints.append(np.array(objtemp))
 
	# if not (charuco_ids is None) and len(charuco_ids) > 0:
	# 		cv.aruco.drawDetectedCornersCharuco(gray, charuco_corners, charuco_ids)
   
	# gray = cv.resize(gray, (480, 640))
	# cv.imshow(str(len(charuco_ids)), gray)
	# cv.waitKey(500)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

np.save('intrinsic.npy', mtx)
np.save('distCoeff.npy', dist)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )

