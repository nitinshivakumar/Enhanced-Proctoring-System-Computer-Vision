import dlib
import track_gaze
import cv2

detector = dlib.get_frontal_face_detector() # a pre-trained face detection model that is capable of locating faces in images. 
predictor = dlib.shape_predictor("Resources/shape_predictor_68_face_landmarks.dat") # create an instance of the shape_predictor class in dlib, using the file path

def get_face_location(start, end, face, gray):
    """
    Returns the coordinates of the face features.

    Inputs:
    start - starting point in the locations of 68 such landmarks on a given face image to get the face features.
    end - ending point in the locations of 68 such landmarks on a given face image to get the face features.
    face - an array that contains the face.
    gray - gray scale of the face.
    
    Outputs:
    location - return the coordinates of the face features.
    """
    landmarks = predictor(gray, face)
    location = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(start, end)]
    return location
    
def get_eye_threshold(image):
    """
    Computes the eye threshold of the student on where he is looking.

    Inputs:
    image - an array which contains student image.
    
    Outputs:
    eye_location_width - the eye threshold of the student .
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        right_eye = get_face_location(36, 42, face, gray)
        left_eye = get_face_location(42, 48, face, gray)
        eye_location_width_right = track_gaze.eye_track_threshold(gray, right_eye)
        eye_location_width_left = track_gaze.eye_track_threshold(gray, left_eye)
        if eye_location_width_right and eye_location_width_left:
            eye_location_width = (eye_location_width_right + eye_location_width_left)/2
            return eye_location_width

def check_eye_fraudalent(gray, left_eye_threshold, right_eye_threshold):
    """
    Check if there is fraudalent detected in the eye tracking.

    Inputs:
    gray - gray scale image of the student.
    left_eye_threshold - threshold set to the left side.
    right_eye_threshold - threshold set to the right side.
    
    Outputs:
    (bool) - true/false on the eye tracking fraudalent.
    """
    thresold = get_eye_threshold(gray)
    print(thresold)
    if thresold != None:
        if (thresold  <= left_eye_threshold) and (thresold >= right_eye_threshold):
            return False
        else:
            return True
    else:
        return True
      
        