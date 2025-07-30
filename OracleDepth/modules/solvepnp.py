import re
import numpy as np
import cv2 as cv
import glob

_defaultRefParams = cv.aruco.RefineParameters(minRepDistance = 10.,
                                           errorCorrectionRate = 3.,
                                           checkAllOrders = True
                                            )

_defaultCharucoParams = cv.aruco.CharucoParameters()
_defaultCharucoParams.tryRefineMarkers = True

def getPos(objImagesLoc, mtx, dist, charucoBoard: cv.aruco.CharucoBoard, charucoParams : cv.aruco.CharucoParameters = _defaultCharucoParams, refParams : cv.aruco.RefineParameters = _defaultRefParams):
    images = glob.glob(f'{objImagesLoc}/*.jpg')
    charuco_detector = cv.aruco.CharucoDetector(charucoBoard, refineParams=refParams, charucoParams=charucoParams)

    rvecs = []
    tvecs = []
    pnpindex = []

    for fname in images:
        img = cv.imread(fname)

        regex = r'(\d+)(?=\.jpg$)'
        i = int(re.search(regex, str(fname)).group(1))
        
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(img)

        objPoints, imgPoints = charucoBoard.matchImagePoints(charuco_corners, charuco_ids)
        ret, rvec, tvec, _ = cv.solvePnPRansac(objPoints, imgPoints, mtx, dist)
        rvec, tvec = cv.solvePnPRefineVVS(objPoints, imgPoints, mtx, dist, rvec, tvec)

        rvecs.append(rvec)
        tvecs.append(tvec)
        pnpindex.append(i)

    return pnpindex, rvecs, tvecs



