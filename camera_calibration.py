import cv2
import setup_calibration 
import eye_motion_tracking

def main(numberOfCalibrations = 3):
    """
    Computes the eye calibration for the test taker.

    Inputs:
    numberOfCalibrations - number of calibration measurements.
    
    Outputs:
    l - computed left side of calibration.
    r - computed right side of calibration.
    """
    cap = cv2.VideoCapture(0)
    cap.set(3, 1080)
    cap.set(4, 1080)

    r1 = []
    l1 = []
    new_setup = ['', '_up', '_down']

    for i in range(numberOfCalibrations):
        r1.append(eye_motion_tracking.get_eye_threshold(setup_calibration.run_webcam('right' + new_setup[i])))
        l1.append(eye_motion_tracking.get_eye_threshold(setup_calibration.run_webcam('left' + new_setup[i])))
        if numberOfCalibrations == i+1:
            setup_calibration.run_webcam('stop')

    print(r1)
    l = max(l1) + 0.05
    r = min(r1) - 0.05

    return l, r






