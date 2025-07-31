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
# _defaultCharucoParams.minMarkers = 1

_defaultdetparams = cv.aruco.DetectorParameters()
_defaultdetparams.aprilTagDeglitch = 1
_defaultdetparams.cornerRefinementMethod = cv.aruco.CORNER_REFINE_APRILTAG
_defaultdetparams.useAruco3Detection = True

def getPos(objImagesLoc, mtx, dist, charucoBoard: cv.aruco.CharucoBoard, charucoParams : cv.aruco.CharucoParameters = _defaultCharucoParams, refParams : cv.aruco.RefineParameters = _defaultRefParams, detparams : cv.aruco.DetectorParameters = _defaultdetparams):
    images = glob.glob(f'{objImagesLoc}/*.jpg')
    charuco_detector = cv.aruco.CharucoDetector(charucoBoard, refineParams=refParams, charucoParams=charucoParams, detectorParams=detparams)

    rvecs = []
    tvecs = []
    pnpindex = []
    errors = []
    count = 0

    for fname in images:
        img = cv.imread(fname)
        img = cv.undistort(img, mtx, dist)

        regex = r'(\d+)(?=\.jpg$)'
        i = int(re.search(regex, str(fname)).group(1))
        rvec, tvec = None, None

        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(img)
        error = 9999

        try:
            objPoints, imgPoints = charucoBoard.matchImagePoints(charuco_corners, charuco_ids)
            ret, rvec, tvec = cv.solvePnP(objPoints, imgPoints, mtx, dist, flags=cv.SOLVEPNP_SQPNP)
            rvec, tvec = cv.solvePnPRefineVVS(objPoints, imgPoints, mtx, dist, rvec, tvec)

            #calc reproj values
            reprojected_points, _ = cv.projectPoints(objPoints, rvec, tvec, mtx, dist)

            imgPoints = np.array(imgPoints, dtype=np.float32).reshape(-1, 2)
            reprojected_points = reprojected_points.reshape(-1, 2)

            error = cv.norm(imgPoints, reprojected_points, cv.NORM_L2) / len(imgPoints)
            
        except:
            # raise ValueError(f'There is an error in image: {fname}')
            print(f'Couldnt solvePnP on {i}')
            count+=1

        errors.append(error)
        rvecs.append(rvec)
        tvecs.append(tvec)
        pnpindex.append(i)

    print(f'Total images with failed solvePnP: {count}')
    return pnpindex, rvecs, tvecs, errors



