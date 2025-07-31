import glob
import json
import re
import cv2 as cv
import os
import numpy as np

def genViews(cwd, metadata):
    images = glob.glob(f'{cwd}\\objImgs\\*.jpg')

    views = []
    for fname in images:
        img = cv.imread(fname)

        regex = r'(\d+)(?=\.jpg$)'
        i = int(re.search(regex, str(fname)).group(1))

        viewId = i
        poseId = i
        intrinsicId = 1453
        path = os.path.abspath(fname)
        path = path.replace("\\", "\\/")
        height, width = img.shape[:2]

        view = {
                'viewId': str(viewId),
                'poseId': str(poseId),
                # 'frameId': frameId,
                'intrinsicId': str(intrinsicId),
                # 'resectionId': resectionId,
                'path': path,
                'width': str(width),
                'height': str(height),
                'metadata': metadata
            }
        
        views.append(view)

    return views

def genIntrinsics(mtx, dist, sensorWidth, sensorHeight, serialNumber, width, height):
    intrinsicId = 1453
    
    mtx = mtx * (sensorWidth/width)
    focalLength, principalPoint = _convMtx(mtx)
    dist = dist[0]

    intrinsics = {
                    'intrinsicId': str(intrinsicId),
                    'width': str(int(width)),
                    'height': str(int(height)),
                    'sensorWidth': str(sensorWidth),
                    'sensorHeight': str(sensorHeight),
                    'serialNumber': serialNumber,
                    'type': 'radial3',
                    'initializationMode': 'calibrated',
                    'initialFocalLength': '-1',
                    'focalLength': str(focalLength),
                    'pixelRatio': '1',
                    'pixelRatioLocked': 'true',
                    'principalPoint': [
                        str(principalPoint[0]),
                        str(principalPoint[1])
                    ],
                    'distortionInitializationMode': 'calibrated',
                    'distortionParams': [
                        str(dist[0]),
                        str(dist[1]),
                        str(dist[4])
                        # str(dist[3]), only 5 because no tangential distortion in radial3
                        # str(dist[4])
                    ],
                    "undistortionOffset": [
                        "0",
                        "0"
                    ],
                    "undistortionParams": "",
                    'locked': 'true'
                }
    
    return [intrinsics]

def _convMtx(mtx):
    focalLength = (mtx[0][0] + mtx[1][1]) / 2.0
    principalPoint = [mtx[0][2], mtx[1][2]]

    return focalLength, principalPoint

def genPoses(pnpindex, rvecs, tvecs, errors):
    poses = []
    
    for i in range(len(pnpindex)):
        p = pnpindex[i]
        r = rvecs[i]
        t = tvecs[i]
        error = errors[i]
        
        maxReprojError = 1
        if not (error > 0 and error < maxReprojError):
            continue

        if r is None or t is None:
            continue

        C = -cv.Rodrigues(r)[0].T @ t
        C = C.flatten()
        
        rot_mtx = _convRot(r)

        pose = {
                'poseId': str(p),
                'pose': {
                    'transform': {
                        'rotation': [
                            str(rot_mtx[0]),
                            str(rot_mtx[1]),
                            str(rot_mtx[2]),
                            str(rot_mtx[3]),
                            str(rot_mtx[4]),
                            str(rot_mtx[5]),
                            str(rot_mtx[6]),
                            str(rot_mtx[7]),
                            str(rot_mtx[8])
                        ],
                        'center': [
                            str(C[0]),
                            str(C[1]),
                            str(C[2])
                        ]
                    },
                    'locked': '1'
                }
            }

        poses.append(pose)

    return poses

def _convRot(r):
    rot_mtx, _ = cv.Rodrigues(r)

    return rot_mtx.T.flatten()

def genSfMDataMethodB(cwd, mtx, dist, metadata, sensorWidth, sensorHeight, serialNumber, width, height, pnpindex, rvecs, tvecs, errors):
    views = genViews(cwd, metadata)
    intrinsics = genIntrinsics(mtx, dist, sensorWidth, sensorHeight, serialNumber, width, height)
    poses = genPoses(pnpindex, rvecs, tvecs, errors)

    data = {
        'version': ['1', '2', '6'],
        'views': views,
        'intrinsics': intrinsics,
        'poses': poses
        }
    
    with open(f'{cwd}\\results\\cameraData.sfm', 'w') as f:
        json.dump(data, f, indent=4)

def genSfMDataMethodC(cwd, mtx, dist, metadata, sensorWidth, sensorHeight, serialNumber, width, height):
    views = genViews(cwd, metadata)
    intrinsics = genIntrinsics(mtx, dist, sensorWidth, sensorHeight, serialNumber, width, height)

    data = {
        'version': ['1', '2', '6'],
        'views': views,
        'intrinsics': intrinsics
        }
    
    with open(f'{cwd}\\results\\cameraData.sfm', 'w') as f:
        json.dump(data, f, indent=4)
