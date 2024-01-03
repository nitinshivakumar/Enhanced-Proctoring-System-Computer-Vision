import cv2
import face_recognition
import os

def encode_generator():
    new_image_list = [item 
                      for item in os.listdir('FaceEncode/Images') 
                      if not item.startswith('.') and 
                      os.path.isfile(os.path.join('FaceEncode/Images', item))]
    new_image_list = ['FaceEncode/Images/' + image for image in new_image_list]
    return new_image_list

def find_encodings(image_list):
    encode_list = []
    for i in image_list:
        current_image = cv2.imread(i)
        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(current_image)[0]
        encode_list.append(encode)
    return encode_list   


    
