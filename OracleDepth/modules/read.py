import json

def readInitFile(cwd):
    with open(f'{cwd}\\results\\cameraInit.sfm', 'r') as f:
        data = json.load(f)

        metadata = data['views'][0]['metadata']
        sensorWidth = float(data['intrinsics'][0]['sensorWidth'])
        sensorHeight = float(data['intrinsics'][0]['sensorHeight'])
        serialNumber = data['intrinsics'][0]['serialNumber']
        width = float(data['intrinsics'][0]['width'])
        height = float(data['intrinsics'][0]['height'])
        
    return metadata, sensorWidth, sensorHeight, serialNumber, width, height