'''
input any video of a person in .mp4 and choose the rgb channel you want to extract
output is the mean of each frame of the rgb channel after masking only the cheeks of the person
'''

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def download_model_if_missing(model_path="face_landmarker.task"):
    model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    if not os.path.exists(model_path):
        print("Downloading face_landmarker.task model...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Download complete!")

def extract_upper_cheeks_rgb(video_path: str, model_path: str = "face_landmarker.task"):
    """
    Tracks upper cheek masks (left + right) and calculates average R, G, and B channel
    values for each frame across the combined cheek regions.

    Returns: r_means, g_means, b_means: 1D numpy arrays containing mean values per frame.
    """

    #run model on video for 1 face
    download_model_if_missing(model_path)
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,running_mode=vision.RunningMode.VIDEO, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    #upper cheek landmarks
    LEFT_UPPER_CHEEK = [117, 118, 101, 36, 205, 207, 187, 147, 123]
    RIGHT_UPPER_CHEEK = [346, 347, 330, 266, 425, 427, 411, 376, 352]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None, None

    #get information about the video file
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_timestamp_ms = 0
    frame_duration_ms = int(1000 / fps)

    # Buffers for temporal smoothing
    prev_left_pts = None
    prev_right_pts = None
    smooth_factor = 0.6

    # Output vectors
    r_means, g_means, b_means = [], [], []


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += frame_duration_ms

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]

            def get_raw_pts(landmark_indices):
                pts = []
                for idx in landmark_indices:
                    lm = landmarks[idx]
                    pts.append([int(lm.x * width), int(lm.y * height)])
                return np.array(pts, dtype=np.float32)

            # Extract raw points of the cheeks
            raw_left_pts = get_raw_pts(LEFT_UPPER_CHEEK)
            raw_right_pts = get_raw_pts(RIGHT_UPPER_CHEEK)

            # Temporal Smoothing (to prevent glitches when the face moves)
            if prev_left_pts is None:
                smooth_left_pts = raw_left_pts
            else:
                #exponntial moving average: position of the point is based on the position of the current frame and the previous frame
                #smooth factor can be adjusted: currently it's 0.6
                smooth_left_pts = (smooth_factor * raw_left_pts) + ((1 - smooth_factor) * prev_left_pts)
            prev_left_pts = smooth_left_pts

            if prev_right_pts is None:
                smooth_right_pts = raw_right_pts
            else:
                smooth_right_pts = (smooth_factor * raw_right_pts) + ((1 - smooth_factor) * prev_right_pts)
            prev_right_pts = smooth_right_pts

            #convert to integer pixel coordinates
            left_int = smooth_left_pts.astype(np.int32)
            right_int = smooth_right_pts.astype(np.int32)

            #convex hulls to guarantee non-intersecting shapes
            left_hull = cv2.convexHull(left_int)
            right_hull = cv2.convexHull(right_int)

            # Create binary mask for both cheeks (255 inside cheeks, 0 outside)
            cheeks_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(cheeks_mask, [left_hull], 255)
            cv2.fillPoly(cheeks_mask, [right_hull], 255)

            # Calculate mean BGR values strictly within the cheek regions
            # OpenCV frames are BGR by default, we need to convert to RGB
            b_mean, g_mean, r_mean, _ = cv2.mean(frame, mask=cheeks_mask)

            r_means.append(r_mean)
            g_means.append(g_mean)
            b_means.append(b_mean)
        else:
            # If no face is detected, append 0
            r_means.append(0)
            g_means.append(0)
            b_means.append(0)

    cap.release()
    detector.close()

    print(f"Extraction complete! Extracted {len(r_means)} frame measurements.")
    return np.array(r_means), np.array(g_means), np.array(b_means)


def signal_extraction_main_cheeks(input_video, input_channel):
  '''
  input_video is a video file .mp4
  input_chanel is either:
  'r' for red channel,'g' for green channel or 'b' for blue channel
  '''

  # Run signal extraction
  r_vector, g_vector, b_vector= extract_upper_cheeks_rgb(input_video)
  if input_channel == 'r':
      return r_vector
  elif input_channel == 'g':
      return g_vector
  elif input_channel == 'b':
      return b_vector
  else:
      print('invalid input channel')