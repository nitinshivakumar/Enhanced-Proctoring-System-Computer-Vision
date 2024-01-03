import cv2
import numpy as np
import face_recognition
import copy
import eye_motion_tracking as eye
import camera_calibration as cc
import FaceEncode.face_encode as face_encode
import mouth
from ultralytics import YOLO

# Initializing all the requierd parameters
cap = cv2.VideoCapture(0)
success, img = cap.read()
cap.set(3, 1080)
cap.set(4, 1080)
eye_fault_counter = 0
track_eye_fraud = 0
suspicious_count = 0
speak = 0

#Generate the student image encoding who are allowed to take exam
new_image_list  = face_encode.encode_generator()
print("Encoding commeneced . . . . . .")
face_encode_list = face_encode.find_encodings(new_image_list)
print("Encoding complete")


#To detect faces, load the pre-trained Haar Cascade classifier.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Load the necessary images.
imgBackground = cv2.imread('Resources/UB.png')
originalBackground = copy.deepcopy(imgBackground)
fraudBackground = copy.deepcopy(imgBackground)
imgWarning = cv2.imread('Resources/warning.png')
imgFraud = cv2.imread('Resources/fraud_detect.png')
mobile_image = cv2.imread('Resources/mobile.png')
speak_image = cv2.imread('Resources/speak.png')
eye_image = cv2.imread('Resources/eye.png')
student_image = cv2.imread('Resources/student.png')
fraudBackground[277:277+imgFraud.shape[0], 185:185+imgFraud.shape[1]] = imgFraud

#Set the threshold for the various fradualencies
max_eye_fault = 10
max_eye_warning = 1
max_suspicious_count = 6
mouth_thres = 4

def eye_moment(gray_image, eye_fault_counter, imgBackground):
    """
    Check whether there is a fault in eye.

    Inputs:
    gray_image - a matrix containing gray image
    eye_fault_counter - the variable that counts the eye fault
    imgBackground - the matrix containing background and real time image.
    
    Outputs:
    eye_fault_counter - the variable that counts the eye fault
    imgBackground - the matrix containing background and real time image.
    """
    if eye.check_eye_fraudalent(gray_image, left_eye_thresold, right_eye_thresold):
        eye_fault_counter += 1
        imgBackground[393: 393 + eye_image.shape[0],
                      1503: 1503 + eye_image.shape[1]] = eye_image
    else:
        eye_fault_counter = 0
        imgBackground[393: 393 + eye_image.shape[0],
                      1503: 1503 + eye_image.shape[1]] = originalBackground[393: 393 + eye_image.shape[0],
                                                                                           1503: 1503 + eye_image.shape[1]]
    return eye_fault_counter, imgBackground

def draw_rectangle(face_location):
    """
    Calculates the rectangle dimensions given the face location.

    Inputs:
    face_location - distances array represents the Euclidean distance between the target face encoding (encodeFace).
    
    Outputs:
    face_box - Calculates the dimension of the rectangle.
    """
    y1, x2, y2, x1 = face_location
    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
    face_box = 185 + x1, 277 + y1, x2 - x1, y2 - y1
    return face_box

def check_on_speaking(mouth_threshold, min_speak):
    """
    Check whether the student is speaking.

    Inputs:
    mouth_threshold - the distance between the lips.
    min_speak - the variable that counts the mouth fault.
    
    Outputs:
    min_speak - the variable that counts the mouth fault.
    """
    if mouth_threshold:
        if mouth_threshold > mouth_thres:
            min_speak += 1
            return min_speak
        return 0
    else:
        return 0

def suspicious_item_detected(img):
    """
    Check whether there is a mobile detect.

    Inputs:
    img - a matrix containing live image.
    
    Outputs:
    (bool) - true/false for mobile detect.
    """
    model = YOLO('best.pt')
    results = model.predict(img)
    
    for r in results:
        boxes = r.boxes
        if len(boxes):
            print('Mobile Phone detected')
            return True
    return False

suspicious_item_detected(img)
mask_background = np.ones_like(imgBackground)

#Calibration
left_eye_thresold, right_eye_thresold = cc.main()
print(left_eye_thresold, right_eye_thresold)

while True:
    success, img = cap.read()
    if not success:
        print("Problem in detecting the camera.")
        break
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Perform face detection
    faces = face_cascade.detectMultiScale(gray_image,
                                           scaleFactor=1.1,
                                             minNeighbors=5,
                                               minSize=(30, 30))
    
    #Set initial speaking as false.
    talking = False
    
    #Check if there is any exceeding of the maximum fraud with respect to the eye and mobile detect.
    if (track_eye_fraud < max_eye_warning) or (suspicious_count > max_suspicious_count):

        #Check if the numer of faces detected.
        if len(faces) > 0:

            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25) # Resize the image.
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB) # Change the BGR to RGB image.
            current_face_frame = face_recognition.face_locations(imgS) # Get the current face location
            encode_current_face = face_recognition.face_encodings(imgS, current_face_frame) # Get the students allowed face encodings.
            imgBackground[277:277+img.shape[0], 185:185+img.shape[1]] = img # Set the webcam to the UB background.

            for encodeFace, faceLoc in zip(encode_current_face, current_face_frame):
                matches = face_recognition.compare_faces(face_encode_list, encodeFace) # compares a list of face encodings.
                faceDis = face_recognition.face_distance(face_encode_list, encodeFace) #  compute the Euclidean distance between a list of face encodings and a target face encoding.
                print("matches", matches)
                print("faceDis", faceDis)
                match = np.argmin(faceDis) 
                rect_color = (0, 0, 255)

                # Check if there is a match in the face.
                if matches[match] == True:
                    print("Student allowed to take test")
                    imgBackground[993: 993 + imgWarning.shape[0],
                                   1503: 1503 + imgWarning.shape[1]] = originalBackground[993: 993 + imgWarning.shape[0],
                                                                                            1503: 1503 + imgWarning.shape[1]]
                    
                    eye_fault_counter, imgBackground = eye_moment(img, eye_fault_counter, imgBackground) # Test the eye moment.
                    check_suspicious = suspicious_item_detected(img) # Detect the mobile if present in the surrounding.
                    mouth_threshold = mouth.detect_speak(img) # Get the mouth open thresold.
                    speak = check_on_speaking(mouth_threshold, speak) # Check whether the student is speaking.

                    # Check whether the student is speaking continuously
                    if speak > 3:
                        imgBackground[993: 993 + speak_image.shape[0],
                                1503: 1503 + speak_image.shape[1]] = speak_image

                    # Check if there is mobile detected continously in the background.
                    if check_suspicious:
                        suspicious_count += 1
                        imgBackground[693: 693 + mobile_image.shape[0], 1503: 1503 + mobile_image.shape[1]] = mobile_image
                    else:
                        suspicious_count = 0
                        imgBackground[693: 693 + mobile_image.shape[0],
                                   1503: 1503 + mobile_image.shape[1]] = originalBackground[693: 693 + mobile_image.shape[0],
                                                                                           1503: 1503 + mobile_image.shape[1]]

                    # Check whether the student is looking out the window. 
                    if eye_fault_counter > max_eye_fault:
                        eye_fault_counter = 0 # reset eye fault counter
                        imgBackground[993: 993 + imgWarning.shape[0], 1503: 1503 + imgWarning.shape[1]] = imgWarning
                        track_eye_fraud += 1

                else:
                    face_box = draw_rectangle(faceLoc)
                    imgBackground = cv2.rectangle(imgBackground, 
                                                  face_box, 
                                                  rect_color, 
                                                  thickness=2)
                    imgBackground[993: 993 + imgWarning.shape[0], 1503: 1503 + imgWarning.shape[1]] = imgWarning
                    print("Fraudulent student")

        else:
            print('no student')
            imgBackground[297: 297 + student_image.shape[0], 637: 637 + student_image.shape[1]] = student_image
            if track_eye_fraud >= max_eye_warning:
                imgBackground = originalBackground

    else:
        cv2.imwrite('FaceEncode/FraudImage/fraud_image.png', img)
        imgBackground = fraudBackground
        break

    cv2.imshow("Enhanced Proctoring", imgBackground)
    cv2.waitKey(1)