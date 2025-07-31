# OracleDepth

## Requirements
- Windows 11
- CUDA able GPU
- Python
---

## Installation & Setup

### 1. Clone the Repository

    git clone https://github.com/yourusername/OracleDepth.git
    cd OracleDepth

---

### 2. Install Python Dependencies

    pip install -r requirements.txt

---

### 3. Download and Set Up Meshroom

- Download Meshroom from the official website:  
  https://alicevision.org/#meshroom

- Recommended version: Meshroom-2023.3.0

- After downloading:
  - Place the folder inside the main `OracleDepth/` directory.
  - Rename the folder to `Meshroom`

---

### 4. Insert Images

- Place object images into the `objImgs/` folder.
- For methods A, B, and C, also insert calibration images into the `calibImgs/` folder.

Example folder structure:

    OracleDepth
    ├── Meshroom
    ├── objImgs
    │   └── [your object images here]
    ├── calibImgs
    │   └── [your calibration images here, if needed]
    ├── results

---

### 5. Run the Application

    python main.py

---

## Explaining the methods
- Method A(WIP) - Calibrates the camera based on the calibration images and uses that information along with user given positions to reconstructs and scales the object from the object images allowing for precise measurements to be done on the object.
- Method B - Calibrates the camera with the calibration images and using solvePnP gets the position of the camera in the object images and using the calibration data and the best positions of cameras, reconstructs and scales the object allowing for precise measurements to be done on the object.
- Method C - Calibrates the camera with the calibration images and reconstructs the object from the object images.
- Method D - Reconstructs the object from the object images.

Intrinsic parameters(method D) and positions(method C and D) can be aproximated by the StructureFromMotion node in Meshroom, but for better reconstruction and for precise scaling we can import the positions of the cameras(method A, B).
