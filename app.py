import streamlit as st
import tempfile
import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import csv
from tqdm import tqdm
import base64
from pathlib import Path

def plot_world_landmarks(plt, ax, landmarks, visibility_th=0.5):
    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_point.append([landmark.visibility, (landmark.x, landmark.y, landmark.z)])

    face_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    right_arm_index_list = [11, 13, 15, 17, 19, 21]
    left_arm_index_list = [12, 14, 16, 18, 20, 22]
    right_body_side_index_list = [11, 23, 25, 27, 29, 31]
    left_body_side_index_list = [12, 24, 26, 28, 30, 32]
    shoulder_index_list = [11, 12]
    waist_index_list = [23, 24]

    # Face
    face_x, face_y, face_z = [], [], []
    for index in face_index_list:
        point = landmark_point[index][1]
        face_x.append(point[0])
        face_y.append(point[2])
        face_z.append(point[1] * (-1))

    # Right arm
    right_arm_x, right_arm_y, right_arm_z = [], [], []
    for index in right_arm_index_list:
        point = landmark_point[index][1]
        right_arm_x.append(point[0])
        right_arm_y.append(point[2])
        right_arm_z.append(point[1] * (-1))

    # Left arm
    left_arm_x, left_arm_y, left_arm_z = [], [], []
    for index in left_arm_index_list:
        point = landmark_point[index][1]
        left_arm_x.append(point[0])
        left_arm_y.append(point[2])
        left_arm_z.append(point[1] * (-1))

    # Right body side
    right_body_side_x, right_body_side_y, right_body_side_z = [], [], []
    for index in right_body_side_index_list:
        point = landmark_point[index][1]
        right_body_side_x.append(point[0])
        right_body_side_y.append(point[2])
        right_body_side_z.append(point[1] * (-1))

    # Left body side
    left_body_side_x, left_body_side_y, left_body_side_z = [], [], []
    for index in left_body_side_index_list:
        point = landmark_point[index][1]
        left_body_side_x.append(point[0])
        left_body_side_y.append(point[2])
        left_body_side_z.append(point[1] * (-1))

    # Shoulder
    shoulder_x, shoulder_y, shoulder_z = [], [], []
    for index in shoulder_index_list:
        point = landmark_point[index][1]
        shoulder_x.append(point[0])
        shoulder_y.append(point[2])
        shoulder_z.append(point[1] * (-1))

    # Waist
    waist_x, waist_y, waist_z = [], [], []
    for index in waist_index_list:
        point = landmark_point[index][1]
        waist_x.append(point[0])
        waist_y.append(point[2])
        waist_z.append(point[1] * (-1))
            
    ax.cla()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    ax.scatter(face_x, face_y, face_z)
    ax.plot(right_arm_x, right_arm_y, right_arm_z)
    ax.plot(left_arm_x, left_arm_y, left_arm_z)
    ax.plot(right_body_side_x, right_body_side_y, right_body_side_z)
    ax.plot(left_body_side_x, left_body_side_y, left_body_side_z)
    ax.plot(shoulder_x, shoulder_y, shoulder_z)
    ax.plot(waist_x, waist_y, waist_z)
    
    return

def process_video(input_video_path, output_video_path, output_csv_path='landmarks.csv', fps=None, progress_bar=None, pose_params=None):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=pose_params.get('static_image_mode', False),
        model_complexity=pose_params.get('model_complexity', 1),
        smooth_landmarks=pose_params.get('smooth_landmarks', True),
        min_detection_confidence=pose_params.get('min_detection_confidence', 0.5),
        min_tracking_confidence=pose_params.get('min_tracking_confidence', 0.5)
    )

    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        st.error("Error opening video file")
        return False

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps is not None else input_fps if input_fps else 30

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Set up CSV file for saving landmarks
    csv_file = open(output_csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame_number', 'landmark_id', 'x', 'y', 'z', 'visibility'])

    # Set up Matplotlib figure
    plt.ioff()  # Turn off interactive mode for Streamlit compatibility
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize progress tracking
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Clear previous plot
        ax.cla()
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if results.pose_world_landmarks:
            # Plot landmarks
            plot_world_landmarks(plt, ax, results.pose_world_landmarks)
            
            # Save landmarks to CSV
            for idx, landmark in enumerate(results.pose_world_landmarks.landmark):
                csv_writer.writerow([
                    frame_number,
                    idx,
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    landmark.visibility
                ])
        else:
            # Write empty entry for missing detection
            csv_writer.writerow([frame_number, -1, None, None, None, None])

        # Convert plot to image
        fig.canvas.draw()
        pose_img = np.array(fig.canvas.buffer_rgba())
        pose_img = cv2.cvtColor(pose_img, cv2.COLOR_RGBA2BGR)
        pose_img = cv2.resize(pose_img, (width, height))

        # Overlay visualization
        combined_frame = cv2.addWeighted(frame, 0.7, pose_img, 0.3, 0)
        video_writer.write(combined_frame)

        frame_number += 1
        
        # Update progress bar
        if progress_bar is not None:
            progress_bar.progress(frame_number / total_frames)

    # Cleanup resources
    cap.release()
    video_writer.release()
    csv_file.close()
    plt.close(fig)
    return True

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(bin_file)}">{file_label}</a>'
    return href

def app():
    st.title("3D Pose Estimation Video Processor")
    st.write("Upload a video file to generate 3D pose estimation visualization")
    
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    
    col1, col2 = st.columns(2)
    with col1:
        output_filename = st.text_input("Output filename (without extension)", "output")
    
    with col2:
        fps = st.number_input("Output FPS (leave at 0 to use source FPS)", value=0, min_value=0, max_value=60)
        fps = None if fps == 0 else fps
    
    generate_csv = st.checkbox("Generate CSV file with landmark data", value=True)
    
    # Add advanced options in an expander
    with st.expander("Advanced Options"):
        st.subheader("MediaPipe Pose Parameters")
        
        # Create columns for better organization
        col1, col2 = st.columns(2)
        
        with col1:
            static_image_mode = st.checkbox("Static Image Mode", value=False)
            model_complexity = st.selectbox(
                "Model Complexity",
                options=[0, 1, 2],
                index=1,
                help="Higher complexity means better accuracy but slower processing"
            )
            smooth_landmarks = st.checkbox("Smooth Landmarks", value=True)
        
        with col2:
            min_detection_confidence = st.slider(
                "Minimum Detection Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Higher values mean more confident detections but might miss some poses"
            )
            min_tracking_confidence = st.slider(
                "Minimum Tracking Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Higher values mean more confident tracking but might lose tracking more easily"
            )
        
        # Store parameters in a dictionary
        pose_params = {
            'static_image_mode': static_image_mode,
            'model_complexity': model_complexity,
            'smooth_landmarks': smooth_landmarks,
            'min_detection_confidence': min_detection_confidence,
            'min_tracking_confidence': min_tracking_confidence
        }
    
    if uploaded_file is not None:
        st.video(uploaded_file)
        
        if st.button("Process Video"):
            # Create temp directory
            temp_dir = tempfile.mkdtemp()
            
            # Save uploaded file to temp directory
            input_path = os.path.join(temp_dir, "input_video.mp4")
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Set output paths
            output_video_path = os.path.join(temp_dir, f"{output_filename}.mp4")
            output_csv_path = os.path.join(temp_dir, f"{output_filename}.csv") if generate_csv else None
            
            # Process video with progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Processing video... This may take a while depending on the video length.")
            
            try:
                success = process_video(input_path, output_video_path, output_csv_path, fps, progress_bar, pose_params)
                
                if success:
                    status_text.text("Processing complete!")
                    
                    # Display download links
                    st.subheader("Download Results")
                    
                    # Video download
                    video_link = get_binary_file_downloader_html(output_video_path, f"Download processed video ({output_filename}.mp4)")
                    st.markdown(video_link, unsafe_allow_html=True)
                    
                    # CSV download (if generated)
                    if generate_csv and os.path.exists(output_csv_path):
                        csv_link = get_binary_file_downloader_html(output_csv_path, f"Download landmark data ({output_filename}.csv)")
                        st.markdown(csv_link, unsafe_allow_html=True)
                else:
                    st.error("Error processing video. Please try a different file.")
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")

if __name__ == "__main__":
    app()