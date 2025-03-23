# Multi-Camera 3D Cricket Pose Estimation

## Introduction
This project aims to accurately estimate 3D poses of cricket players in real-time using multiple synchronized cameras. Leveraging known camera calibration matrices and state-of-the-art 2D keypoint extraction techniques, our solution reconstructs detailed 3D joint positions. The project is designed for deployment in cricket stadiums, where scalability and robust performance are crucial.

In addition to that we have made a cool front end which can be hosted and deployed seamlessly.



## Papers & Projects Referred
The project draws inspiration from several pioneering works:

- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- [MetaPose](https://metapose.github.io/)
- [Faster VoxelPose](https://github.com/AlvinYH/Faster-VoxelPose)
- [MMPose](https://github.com/open-mmlab/mmpose)
- [MVPose](https://github.com/zju3dv/mvpose)
- [TEMPO: Efficient Multi-View Pose Estimation, Tracking, and Forecasting](https://github.com/rccchoudhury/tempo?tab=readme-ov-file)
- [Fast-3D-Human-Pose-Estimation](https://github.com/eddie0509tw/Fast-3D-Human-Pose-Estimation?tab=readme-ov-file)
- [BlazePose](https://github.com/geaxgx/depthai_blazepose)


These references have influenced our approach for 2D keypoint extraction and 3D triangulation.

## Solution Pipeline

1. **Data PreProcessing**  
   - Convert image data (e.g., JPEG sequences) into video format (MP4) for synchronized processing.
  
2. **2D Keypoint Extraction (Keystoning)**  
   - Utilize Mediapipe to detect and extract 2D keypoints from each camera view.
  
3. **3D Triangulation**  
   - Reconstruct 3D joint positions via triangulation, leveraging either classical translation matrices or deep learning-based inference methods.
  
4. **Rendering**  
   - Visualize the reconstructed 3D poses using Matplotlib, Axes3D, and Poly3D.
   - *Note:* Integration with advanced rendering engines like Unreal Engine is possible for enhanced visualization, though asset costs can be prohibitive.

## How to Run the Project

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/Phoenix1859/hackeza_205
   cd hackeza_205
   ```

2. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Launching:**
   - Launch the app.
   ```sh
   streamlit run app.py
   ```


## Solution

### 1) Data Processing  
**Overview:**  
The initial step involves converting raw images into a standardized video format. Each image is treated as a frame and collated into an MP4 file. This standardization is crucial for ensuring that all subsequent processing steps operate on a uniform data format.

**Details:**  
- **Frame Collation:** Raw images captured from various sources are organized in sequence.
- **Video Encoding:** Using [OpenCV](https://opencv.org/) in Python, the frames are encoded into an MP4 file.
- **Benefits:** A single video file reduces I/O overhead and ensures compatibility with video processing libraries.


---

### 2) 2D Inferencing via YOLO  
**Overview:**  
For player detection and tracking, we employed YOLO (You Only Look Once). This model processes the MP4 file to detect players, generate bounding boxes, and extract 2D keypoints.

**Details:**  
- **Detection:** YOLO processes each frame to detect players with high speed and accuracy.
- **Bounding Boxes & Keypoints:** Detected players are enclosed in bounding boxes, and keypoint detection algorithms extract 2D coordinates (e.g., joints).
- **Tools & Frameworks:** Our pipeline may use implementations like 
**Benefits:**  
- Real-time processing capabilities.
- High accuracy in crowded scenes.
- Compatibility with various deep learning frameworks.

**Example Reference:**
![camera2_output-ezgif com-optimize](https://github.com/user-attachments/assets/89cf8124-ce72-4af5-9c79-d548a6276e4f)


---

### 3) 3D Inferencing 
**Overview:**  
The next stage involves converting 2D keypoints into 3D space. This is achieved by inferring depth from the single MP4 file and applying DL techniques to reconstruct the 3D positions of the players.

**Details:**  
- **Depth Estimation:** Using machine learning models, the depth of each keypoint is inferred from the 2D data.
**Benefits:**  
- Provides a spatially accurate representation of player positions.
- Enhances the analysis of player dynamics and interactions.
![3-Dinference1-ezgif com-optimize](https://github.com/user-attachments/assets/4471b7f8-6301-4a7b-99b1-09ac6ca3a2cf)
![3-Dinference2-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/42d5c029-6238-4b58-8c4c-9cd8f2fc8fe7)

---

### 4) Visualization  
**Overview:**  
After processing, the final step is to visualize the 3D reconstructions. We generate a CSV file containing the keypoints (following the standard 33 keypoint human contour configuration per frame) and use visualization libraries to render the data.

**Details:**  
- **CSV Output:** The processed data is output in a CSV format, making it easy to integrate with various visualization tools.
![final-render-ezgif com-optimize](https://github.com/user-attachments/assets/d86cfc50-16a6-4525-9949-e9a04e9af176)



## Contributors
List team members and specify individual contributions.

