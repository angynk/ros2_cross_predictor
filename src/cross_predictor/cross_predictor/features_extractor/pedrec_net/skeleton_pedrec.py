from enum import Enum


class SKELETON_PEDREC_JOINT(Enum):
    nose = 0
    left_eye = 1
    right_eye = 2
    left_ear = 3
    right_ear = 4
    left_shoulder = 5
    right_shoulder = 6
    left_elbow = 7
    right_elbow = 8
    left_wrist = 9
    right_wrist = 10
    left_hip = 11
    right_hip = 12
    left_knee = 13
    right_knee = 14
    left_ankle = 15
    right_ankle = 16
    hip_center = 17
    spine_center = 18
    neck = 19
    head_lower = 20
    head_upper = 21
    left_foot_end = 22
    right_foot_end = 23
    left_hand_end = 24
    right_hand_end = 25


SKELETON_PEDREC_JOINTS = [
    SKELETON_PEDREC_JOINT.nose,
    SKELETON_PEDREC_JOINT.left_eye,
    SKELETON_PEDREC_JOINT.right_eye,
    SKELETON_PEDREC_JOINT.left_ear,
    SKELETON_PEDREC_JOINT.right_ear,
    SKELETON_PEDREC_JOINT.left_shoulder,
    SKELETON_PEDREC_JOINT.right_shoulder,
    SKELETON_PEDREC_JOINT.left_elbow,
    SKELETON_PEDREC_JOINT.right_elbow,
    SKELETON_PEDREC_JOINT.left_wrist,
    SKELETON_PEDREC_JOINT.right_wrist,
    SKELETON_PEDREC_JOINT.left_hip,
    SKELETON_PEDREC_JOINT.right_hip,
    SKELETON_PEDREC_JOINT.left_knee,
    SKELETON_PEDREC_JOINT.right_knee,
    SKELETON_PEDREC_JOINT.left_ankle,
    SKELETON_PEDREC_JOINT.right_ankle,
    SKELETON_PEDREC_JOINT.hip_center,
    SKELETON_PEDREC_JOINT.spine_center,
    SKELETON_PEDREC_JOINT.neck,
    SKELETON_PEDREC_JOINT.head_lower,
    SKELETON_PEDREC_JOINT.head_upper,
    SKELETON_PEDREC_JOINT.left_foot_end,
    SKELETON_PEDREC_JOINT.right_foot_end,
    SKELETON_PEDREC_JOINT.left_hand_end,
    SKELETON_PEDREC_JOINT.right_hand_end
]

SKELETON_PEDREC = [
    (SKELETON_PEDREC_JOINT.right_shoulder.value, SKELETON_PEDREC_JOINT.right_elbow.value),
    (SKELETON_PEDREC_JOINT.right_elbow.value, SKELETON_PEDREC_JOINT.right_wrist.value),
    (SKELETON_PEDREC_JOINT.left_shoulder.value, SKELETON_PEDREC_JOINT.left_elbow.value),
    (SKELETON_PEDREC_JOINT.left_elbow.value, SKELETON_PEDREC_JOINT.left_wrist.value),
    (SKELETON_PEDREC_JOINT.right_hip.value, SKELETON_PEDREC_JOINT.right_knee.value),
    (SKELETON_PEDREC_JOINT.right_knee.value, SKELETON_PEDREC_JOINT.right_ankle.value),
    (SKELETON_PEDREC_JOINT.left_hip.value, SKELETON_PEDREC_JOINT.left_knee.value),
    (SKELETON_PEDREC_JOINT.left_knee.value, SKELETON_PEDREC_JOINT.left_ankle.value),
    (SKELETON_PEDREC_JOINT.nose.value, SKELETON_PEDREC_JOINT.right_eye.value),
    (SKELETON_PEDREC_JOINT.right_eye.value, SKELETON_PEDREC_JOINT.right_ear.value),
    (SKELETON_PEDREC_JOINT.nose.value, SKELETON_PEDREC_JOINT.left_eye.value),
    (SKELETON_PEDREC_JOINT.left_eye.value, SKELETON_PEDREC_JOINT.left_ear.value),
    (SKELETON_PEDREC_JOINT.hip_center.value, SKELETON_PEDREC_JOINT.left_hip.value),
    (SKELETON_PEDREC_JOINT.hip_center.value, SKELETON_PEDREC_JOINT.right_hip.value),
    (SKELETON_PEDREC_JOINT.hip_center.value, SKELETON_PEDREC_JOINT.spine_center.value),
    (SKELETON_PEDREC_JOINT.spine_center.value, SKELETON_PEDREC_JOINT.neck.value),
    (SKELETON_PEDREC_JOINT.neck.value, SKELETON_PEDREC_JOINT.head_lower.value),
    (SKELETON_PEDREC_JOINT.head_lower.value, SKELETON_PEDREC_JOINT.head_upper.value),
    (SKELETON_PEDREC_JOINT.neck.value, SKELETON_PEDREC_JOINT.left_shoulder.value),
    (SKELETON_PEDREC_JOINT.neck.value, SKELETON_PEDREC_JOINT.right_shoulder.value),
    (SKELETON_PEDREC_JOINT.head_lower.value, SKELETON_PEDREC_JOINT.nose.value),
    (SKELETON_PEDREC_JOINT.left_ankle.value, SKELETON_PEDREC_JOINT.left_foot_end.value),
    (SKELETON_PEDREC_JOINT.right_ankle.value, SKELETON_PEDREC_JOINT.right_foot_end.value),
    (SKELETON_PEDREC_JOINT.left_wrist.value, SKELETON_PEDREC_JOINT.left_hand_end.value),
    (SKELETON_PEDREC_JOINT.right_wrist.value, SKELETON_PEDREC_JOINT.right_hand_end.value)
]