from .pedrec_net.skeleton_pedrec import SKELETON_PEDREC_JOINT

def pedestrian_gaze(skeleton):
    eye_nose_fraction = (skeleton[SKELETON_PEDREC_JOINT.nose.value][0] - skeleton[SKELETON_PEDREC_JOINT.right_eye.value][0])/(
        skeleton[SKELETON_PEDREC_JOINT.left_eye.value][0] - skeleton[SKELETON_PEDREC_JOINT.right_eye.value][0])

    looking = 'NotLooking'
    if skeleton[SKELETON_PEDREC_JOINT.right_eye.value][0] < skeleton[SKELETON_PEDREC_JOINT.nose.value][0]-1:
        if skeleton[SKELETON_PEDREC_JOINT.nose.value][0] < skeleton[SKELETON_PEDREC_JOINT.left_eye.value][0]-1:
            if eye_nose_fraction > 0.3 and eye_nose_fraction < 0.7:
                return 'Looking', 1
    return looking, 0