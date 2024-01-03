import cv2

def run_webcam(side):
    imgBackground = cv2.imread('Resources/UB.png')
    eye = cv2.imread('Resources/instruction.png')
    traingle = cv2.imread('Resources/triangle.png')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    camera_running = True
    
    while True:
        _, frame = cap.read()
        imgBackground[277:277+frame.shape[0], 185:185+frame.shape[1]] = frame
        img_end_width = (277 + frame.shape[0] + 5)
        img_end_height = (185+frame.shape[1])//2

        imgBackground[img_end_width:img_end_width+eye.shape[0], img_end_height:img_end_height+eye.shape[1]] = eye

        temp = imgBackground.shape[0]//2
        background = imgBackground.shape[1] - 85
        traingle_width = traingle.shape[0]
        traingle_height = traingle.shape[1]

        if side == 'left':
            imgBackground[temp:temp+traingle_width, 5:5+traingle_height] = traingle
        elif side == 'right':
            imgBackground[temp:temp+traingle_width, background:background+traingle_height] = traingle
        elif side == 'left_up':
            imgBackground[2:2+traingle_width, 5:5+traingle_height] = traingle
        elif side == 'right_up':
            imgBackground[2:2+traingle_width, background:background+traingle_height] = traingle
        elif side == 'left_down':
            imgBackground[1140:1140+traingle_width, 5:5+traingle_height] = traingle
        elif side == 'right_down':
            imgBackground[1140:1140+traingle_width, background:background+traingle_height] = traingle
        elif side == 'stop':
            cap.release()
            cv2.destroyAllWindows()
            return None
        
        cv2.imshow("Camera", imgBackground)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Let go of the camera and shut off every OpenCV window.
            return frame

        if not camera_running:
            cap.release()
            break

