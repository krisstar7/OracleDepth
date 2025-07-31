import re
import subprocess
import os
import shutil
import cv2 as cv
import OracleDepth as od
import random, string
import json
import vedo
import time

_defaultRefParams = cv.aruco.RefineParameters(minRepDistance = 10.,
                                           errorCorrectionRate = 3.,
                                           checkAllOrders = True
                                            )

_defaultCharucoParams = cv.aruco.CharucoParameters()
_defaultCharucoParams.tryRefineMarkers = True

def methodB(cwd, calibBoard, objBoard):
    calibImagesLoc = f'{cwd}\\calibImgs'
    objImagesLoc = f'{cwd}\\objImgs'

    cmd = [
        "Meshroom\\meshroom_batch.exe",
        "--input", "calibImgs",
        "--output", "results",
        "--pipeline", "pipelines\\getMetadata.mg",
        "--cache", f"{cwd}\\MeshroomCache"
    ]

    subprocess.run(cmd, cwd=cwd)

    metadata, sensorWidth, sensorHeight, serialNumber, width, height = od.readInitFile(cwd)
    
    mtx, dist = od.calibrate(calibImagesLoc, calibBoard, _defaultCharucoParams, _defaultRefParams)
    pnpindex, rvecs, tvecs, errors = od.getPos(objImagesLoc, mtx, dist, objBoard)

    od.genSfMDataMethodB(cwd, mtx, dist, metadata, sensorWidth, sensorHeight, serialNumber, width, height, pnpindex, rvecs, tvecs, errors)


    cmd = [
        "Meshroom\\meshroom_batch.exe",
        "--input", "objImgs",
        "--output", "results",
        "--pipeline", "pipelines\\methodB.mg",
        "--cache", f"{cwd}\\MeshroomCache"
    ]

    subprocess.run(cmd, cwd=cwd)

def methodC(cwd, board):
    calibImagesLoc = f'{cwd}\\calibImgs'

    cmd = [
        "Meshroom\\meshroom_batch.exe",
        "--input", "calibImgs",
        "--output", "results",
        "--pipeline", "pipelines\\getMetadata.mg",
        "--cache", f"{cwd}\\MeshroomCache"
    ]

    subprocess.run(cmd, cwd=cwd)

    metadata, sensorWidth, sensorHeight, serialNumber, width, height = od.readInitFile(cwd)
    
    mtx, dist = od.calibrate(calibImagesLoc, board, _defaultCharucoParams, _defaultRefParams)
    od.genSfMDataMethodC(cwd, mtx, dist, metadata, sensorWidth, sensorHeight, serialNumber, width, height)


    cmd = [
        "Meshroom\\meshroom_batch.exe",
        "--input", "objImgs",
        "--output", "results",
        "--pipeline", "pipelines\\methodC.mg",
        "--cache", f"{cwd}\\MeshroomCache"
    ]

    subprocess.run(cmd, cwd=cwd)

def methodD(cwd):
    cmd = [
        "Meshroom\\meshroom_batch.exe",
        "--input", "objImgs",
        "--output", "results",
        "--pipeline", "pipelines\\methodD.mg",
        "--cache", f"{cwd}\\MeshroomCache",
    ]

    subprocess.run(cmd, cwd=cwd)

def clearCache(cwd):
    cache_path = f'{cwd}\\MeshroomCache'
    if os.path.exists(cache_path):
        decision = input("Clear cache[Y/n]: ").strip() or 'Y'
        if decision == 'Y':
            shutil.rmtree(cache_path)
        else:
            raise Exception('Cache not cleared')

    os.makedirs(cache_path)

def modifyPipelines(cwd):
    methodBPath = f'{cwd}\\pipelines\\methodB.mg'
    methodCPath = f'{cwd}\\pipelines\\methodC.mg'

    #mod methodB
    with open(methodBPath, 'r') as f:
        data = json.load(f)

    data['graph']['FeatureExtraction_1']['inputs']['input'] = f'{cwd}\\results\\cameraData.sfm'

    with open(methodBPath, 'w') as f:
        json.dump(data, f, indent=4)

    #mod methodC
    with open(methodCPath, 'r') as f:
        data = json.load(f)

    data['graph']['FeatureExtraction_1']['inputs']['input'] = f'{cwd}\\results\\cameraData.sfm'

    with open(methodCPath, 'w') as f:
        json.dump(data, f, indent=4)

def renameObjImgs(cwd):
    path = f'{cwd}\\objImgs'

    for i, filename in enumerate(os.listdir(path)):
        old = os.path.join(path, filename)
        newname = str(i) + '.jpg'
        new = os.path.join(path, newname)
        os.rename(old, new)

def scrambleNames(cwd):
    path = f'{cwd}\\objImgs'
    chars = string.ascii_letters + string.digits  
    for i, filename in enumerate(os.listdir(path)):
        old = os.path.join(path, filename)
        newname = ''.join(random.choice(chars) for _ in range(6)) + '.jpg'
        new = os.path.join(path, newname)
        os.rename(old, new)

def checkObjImgs(cwd):
    path = f'{cwd}\\objImgs'
    if len(os.listdir(path)) == 0:
        raise ValueError('No object images')
    
    pattern = re.compile(r'^\d+\.jpg$', re.IGNORECASE)
    for filename in os.listdir(path):
        if not pattern.match(filename):
            scrambleNames(cwd)
            renameObjImgs(cwd)
            break

def checkCalibImgs(cwd):
    path = f'{cwd}\\calibImgs'
    if len(os.listdir(path)) == 0:
        raise ValueError('No calibration images')

def inputObjBoardData(calibBoard):
    decision = input("Is the object board the same as the calibration board[Y/n]: ").strip() or 'Y'
    if decision == 'Y':
        return calibBoard
    else:
        print('Input object board data')
        width = int(input('Width: '))
        height = int(input('Height: '))
        square_len = float(input('Square length[m]: '))
        marker_len = float(input('Marker length[m]: '))
        dict = int(input('Marker dictionary number: '))

        aruco_dict = cv.aruco.getPredefinedDictionary(dict)
        board_size = (width, height)
        board = cv.aruco.CharucoBoard(board_size, square_len, marker_len, aruco_dict)

        return board

def inputCalibBoardData():
    print('Input calibration board data')
    width = int(input('Width: '))
    height = int(input('Height: '))
    square_len = float(input('Square length[m]: '))
    marker_len = float(input('Marker length[m]: '))
    dict = int(input('Marker dictionary number: '))

    aruco_dict = cv.aruco.getPredefinedDictionary(dict)
    board_size = (width, height)
    board = cv.aruco.CharucoBoard(board_size, square_len, marker_len, aruco_dict)

    return board

def viewModel(cwd):
    mesh = vedo.load(f'{cwd}\\results\\texturedMesh.obj', f'{cwd}\\results\\texturedMesh.mtl')
    # mesh.texture(f'{cwd}\\results\\texture_1001.jpg')
    vedo.show(mesh, axes=1)

if __name__ == "__main__":
    cwd = os.getcwd()
    
    clearCache(cwd)
    modifyPipelines(cwd)
    checkObjImgs(cwd)

    print("""Choose method
          1. Method A(WIP)
          2. Method B
          3. Method C
          4. Method D
    """)

    method = input("Method number: ")

    if method == '1':
        print('Method A')
        print("Still WIP, available soon")

    elif method == '2':
        print('Method B')
        checkCalibImgs(cwd)
        calibBoard = inputCalibBoardData()
        objBoard = inputObjBoardData(calibBoard)
        start = time.time()
        methodB(cwd, calibBoard, objBoard)
        stop = time.time()
        print(f'Done in {stop-start:.0f} seconds')
        viewModel(cwd)

    elif method == '3':
        print('Method C')
        checkCalibImgs(cwd)
        board = inputCalibBoardData()
        start = time.time()
        methodC(cwd, board)
        stop = time.time()
        print(f'Done in {stop-start:.0f} seconds')
        viewModel(cwd)

    elif method == '4':
        print('Method D')
        start = time.time()
        methodD(cwd)
        stop = time.time()
        print(f'Done in {stop-start:.0f} seconds')
        viewModel(cwd)
    elif method == '5':
        print('Utils')
        print("""Choose utility
          1. View last 3D model
          2. Generate ChAruCo board(WIP)
        """)

        util = input("Utility number: ")

        if util == '1':
            viewModel(cwd)
        # elif util == '2':
        #     genCharuco()
        else:
            raise ValueError('Unknown input')
        
    else:
        raise ValueError('Unknown input')
    
    os.system("pause")