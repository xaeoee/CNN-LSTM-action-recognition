import cv2
import mediapipe as mp
import numpy as np


def preprocess_landmark(video_path, maxframes=50):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)

    landmark_list = []

    LANDMARK_DIM = 24  # 12개 * (x, y)

    frame_count = 0

    previous_landmark = None

    while cap.isOpened() and frame_count < maxframes:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            selected = []

            for idx in range(11, 23):  # 11~22 포함
                lm = landmarks[idx]
                selected.extend([lm.x, lm.y])  # z, visibility 제외

            landmark_list.append(selected)
            previous_landmark = selected
            frame_count += 1
        elif previous_landmark is not None:
            landmark_list.append(previous_landmark)
            frame_count += 1
        else:
            first_frame_pose_not_detected = np.zeros(LANDMARK_DIM, dtype=np.float32)
            landmark_list.append(first_frame_pose_not_detected)
            frame_count += 1



    # numpy 배열로 변환
    landmark_array = np.array(landmark_list)

    # 부족한 프레임 0으로 패딩
    if landmark_array.shape[0] < maxframes:
        padding = np.zeros((maxframes - landmark_array.shape[0], LANDMARK_DIM))
        landmark_array = np.vstack((landmark_array, padding))

    cap.release()
    pose.close()
    
    return landmark_array

# if __name__ == "__main__":
#     a = preprocess_landmark("/home/jaeyoung/CNN-LSTM-action-recognition/demo_vid/ss.mov")
#     print(a.shape)
#     print(a)