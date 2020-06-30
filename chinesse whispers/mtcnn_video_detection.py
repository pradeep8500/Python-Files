from mtcnn.mtcnn import MTCNN
import face_recognition
import cv2

#stream = cv2.VideoCapture('videos/VID_20191213_162943.mp4')
stream=cv2.VideoCapture(0)
writer = None
while True:
		(grabbed, frame) = stream.read()
		if not grabbed:
			break
		detector = MTCNN()
		faces = detector.detect_faces(frame)
		for result in faces:
			x, y, width, height = result['box']
			cv2.rectangle(frame,(x,y),(x+width,y+height),(0,255,0),2)
		if writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter('output/mtcnn.avi', fourcc, 24,
				(frame.shape[1], frame.shape[0]), True)
		if writer is not None:
			writer.write(frame)
stream.release()
