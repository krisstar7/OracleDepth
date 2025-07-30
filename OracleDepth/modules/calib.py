import numpy as np
import cv2 as cv
import glob
import os

_defaultRefParams = cv.aruco.RefineParameters(minRepDistance = 10.,
                                           errorCorrectionRate = 3.,
                                           checkAllOrders = True
                                            )

_defaultCharucoParams = cv.aruco.CharucoParameters()
_defaultCharucoParams.tryRefineMarkers = True

def calibrate(calibImagesLoc : str, charucoBoard: cv.aruco.CharucoBoard, charucoParams : cv.aruco.CharucoParameters = _defaultCharucoParams, refParams : cv.aruco.RefineParameters = _defaultRefParams):
    corners_all = []
    ids_all = []
    image_size = None

    images = glob.glob(calibImagesLoc + '/*.jpg')

    for iname in images:
        img = cv.imread(iname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        corners, ids, _ = cv.aruco.detectMarkers(gray, charucoBoard.getDictionary())
        _, charuco_corners, charuco_ids = cv.aruco.interpolateCornersCharuco(markerCorners=corners, markerIds=ids, image=gray, board=charucoBoard)

        corners_all.append(charuco_corners)
        ids_all.append(charuco_ids)

        if not image_size:
            image_size = gray.shape[::-1]

    rms, mtx, dist, _, _ = cv.aruco.calibrateCameraCharuco(
        charucoCorners=corners_all,
        charucoIds=ids_all,
        board=charucoBoard,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    return mtx, dist


def saveCalib(calibImagesLoc : str, charucoBoard: cv.aruco.CharucoBoard, charucoParams : cv.aruco.CharucoParameters = _defaultCharucoParams, refParams : cv.aruco.RefineParameters = _defaultRefParams, calibDataName: [str] = ['intrinsic', 'distCoeff']):
    mtx, dist = calibrate(calibImagesLoc, charucoBoard, charucoParams, refParams)

    np.save('results/' + calibDataName[0] + '.npy', mtx)
    np.save('results/' + calibDataName[1] + '.npy', dist)

def loadCalib(calibDataName: [str] = ['intrinsic', 'distCoeff']):
    mtx = np.load('results/' + calibDataName[0] + '.npy')
    dist = np.load('results/' + calibDataName[1] + '.npy')

    return mtx, dist






