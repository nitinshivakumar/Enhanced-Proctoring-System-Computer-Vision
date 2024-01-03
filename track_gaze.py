import cv2
import numpy as np
    
def get_eye_contours(contours):
    """
    Computes the eye moments of the test taker.

    Inputs:
    contours - A list of contours, where each contour is represented as a list of points. 
    
    Outputs:
    contour_x - calculated eye moments of the test taker.
    """

    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea) # locate the largest contour (c) in the region
        try:
            moments = cv2.moments(c)
            contour_x = int(moments['m10'] / moments['m00'])
            return contour_x
        except ZeroDivisionError:
            # Deal with the situation when moments['m00'] = 0 in order to prevent division by zero.
            contour_x = 0
            return contour_x

    

def locate(face_point, position, size):
    """
    Locate the face points of the test taker.

    Inputs:
    face_point - A list of contours, where each contour is represented as a list of points. 
    position - side of the eye.
    size - Chose to get the size of the ROI.
    
    Outputs:
    (var) - ROI of the eye.
    """

    if size == 'max':
        return (max(face_point, key=lambda point_coordinate: point_coordinate[position]))[position]
    else:
        return (min(face_point, key=lambda point_coordinate: point_coordinate[position]))[position]

def eye_track_threshold(gray, eye):
    """
    Computes the threshold of the eye on where it is located.

    Inputs:
    gray - gray scale of the student image.
    eye - ROI of the eye cropped.
    
    Outputs:
    eye_location_width - eye location irrespective of the width.
    """
    
    dim = gray.shape # obtaining the image's dimensions
    mask = np.zeros(dim, dtype=np.uint8)  # creating mask .
    Colour_points = np.array(eye, dtype=np.int32) # converting eyePoints into Numpy arrays.
    cv2.fillPoly(mask, [Colour_points], 255) # Filling the Eyes portion with WHITE color.
    eyeImage = cv2.bitwise_and(gray, gray, mask=mask) # Writing gray image where color is White  in the mask using Bitwise and operator.

    eyeImage[mask == 0] = 255 # other then eye area will black, making it white

    cropedEye = eyeImage[locate(eye, 1, 'min'):locate(eye, 1, 'max'),
                          locate(eye, 0, 'min'):locate(eye, 0, 'max')] # cropping the eye form eyeImage.
    
    eye_threshold = cv2.threshold(cropedEye, 80, 255, cv2.THRESH_BINARY_INV) # putting the eye's threshold into practice.
    eye_threshold = cv2.GaussianBlur(eye_threshold[1], (7, 7), 0) # apply Gaussian blur to an image. 
    contours, _ = cv2.findContours(eye_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cX = get_eye_contours(contours)
    if cX is not None:
        centre_calclation = eye_threshold.shape
        centre_width = centre_calclation[1]
        eye_location_width = cX / (centre_width - 30) # Subtracting by 30 to have large ratio.
        return eye_location_width



