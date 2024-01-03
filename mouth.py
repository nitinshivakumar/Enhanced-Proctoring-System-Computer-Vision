from imutils import face_utils
import dlib
import cv2

def check_mouth_distance(mouth):
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
	A = abs(mouth[2][1] - mouth[-1][1])
	return A

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('Resources/shape_predictor_68_face_landmarks.dat')

# grab the indexes of the facial landmarks for the mouth
(m_start, m_end) = (61, 68)

def detect_speak(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the mouth coordinates, then use the
		# coordinates to compute the mouth aspect ratio
		mouth = shape[m_start:m_end]

		mouth_threshold = check_mouth_distance(mouth)
		return mouth_threshold


		
		