import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True,model_complexity=2,enable_segmentation=True,min_detection_confidence=0.5)


def get_numpy(part_index,results):
    part = np.array((results.pose_landmarks.landmark[part_index].x,
                    results.pose_landmarks.landmark[part_index].y))
    return part

def get_eucl(hand,face):
    lst = []
    for i in face:
        for j in hand:
            lst.append(np.linalg.norm(i - j))
    return min(lst)
            
def get_dist(results):
    left_ear = get_numpy(mp_pose.PoseLandmark.LEFT_EAR,results)
    right_ear = get_numpy(mp_pose.PoseLandmark.RIGHT_EAR,results)
    left_mouth = get_numpy(mp_pose.PoseLandmark.MOUTH_LEFT,results)
    right_mouth = get_numpy(mp_pose.PoseLandmark.MOUTH_RIGHT,results)
    left_eye = get_numpy(mp_pose.PoseLandmark.LEFT_EYE_OUTER,results)
    right_eye = get_numpy(mp_pose.PoseLandmark.RIGHT_EYE_OUTER,results)

    left_wrist = get_numpy(mp_pose.PoseLandmark.LEFT_WRIST,results)
    right_wrist = get_numpy(mp_pose.PoseLandmark.RIGHT_WRIST,results)
    left_pinky = get_numpy(mp_pose.PoseLandmark.LEFT_PINKY,results)
    right_pinky = get_numpy(mp_pose.PoseLandmark.RIGHT_PINKY,results)
    left_index = get_numpy(mp_pose.PoseLandmark.LEFT_INDEX,results)
    right_index = get_numpy(mp_pose.PoseLandmark.RIGHT_INDEX,results)
    left_thumb = get_numpy(mp_pose.PoseLandmark.LEFT_THUMB,results)
    right_thumb = get_numpy(mp_pose.PoseLandmark.RIGHT_THUMB,results)
    
    left_face = [left_ear,left_mouth,left_eye]
    left_hand = [left_wrist,left_pinky,left_index,left_thumb]
    right_face = [right_ear,right_mouth,right_eye]
    right_hand = [right_wrist,right_pinky,right_index,right_thumb]
    
    dist = [get_eucl(left_hand,left_face),get_eucl(right_hand,right_face)]
    
    return min(dist)
       
def get_pose(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        a = get_dist(results)  
    else:
        a = 1
    return a
 
 

    
#   NOSE = 0
#   LEFT_EYE_INNER = 1
#   LEFT_EYE = 2
#   LEFT_EYE_OUTER = 3
#   RIGHT_EYE_INNER = 4
#   RIGHT_EYE = 5
#   RIGHT_EYE_OUTER = 6
#   LEFT_EAR = 7
#   RIGHT_EAR = 8
#   MOUTH_LEFT = 9
#   MOUTH_RIGHT = 10
#   LEFT_SHOULDER = 11
#   RIGHT_SHOULDER = 12
#   LEFT_ELBOW = 13
#   RIGHT_ELBOW = 14
#   LEFT_WRIST = 15
#   RIGHT_WRIST = 16
#   LEFT_PINKY = 17
#   RIGHT_PINKY = 18
#   LEFT_INDEX = 19
#   RIGHT_INDEX = 20
#   LEFT_THUMB = 21
#   RIGHT_THUMB = 22
#   LEFT_HIP = 23
#   RIGHT_HIP = 24
#   LEFT_KNEE = 25
#   RIGHT_KNEE = 26
#   LEFT_ANKLE = 27
#   RIGHT_ANKLE = 28
#   LEFT_HEEL = 29
#   RIGHT_HEEL = 30
#   LEFT_FOOT_INDEX = 31
#   RIGHT_FOOT_INDEX = 32   