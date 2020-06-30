from mtcnn.mtcnn import MTCNN
import face_recognition
import cv2
import imutils
import pickle
from matplotlib import pyplot


data = pickle.loads(open('faces11.pickle', "rb").read())
filename = '1576129814154.JPEG'
image = pyplot.imread(filename)
#image = cv2.imread(filename)
detector = MTCNN()
faces = detector.detect_faces(image)
boxes=[]
for result in faces:
	x, y, width, height = result['box']
	boxes.append((y,x+width,y+height,x))
	cv2.rectangle(image,(x,y),(x+width,y+height),(0,0,255),2)
encodings = face_recognition.face_encodings(image, boxes)
names = []
for encoding in encodings:
	matches = face_recognition.compare_faces(data["encodings"],encoding,tolerance=0.38)
	name = "Unknown"
	if True in matches:
		matchedIdxs = [i for (i, b) in enumerate(matches) if b]
		counts = {}
		for i in matchedIdxs:
			name = data["names"][i]
			counts[name] = counts.get(name, 0) + 1
		print(counts)
		name = max(counts, key=counts.get)
	names.append(name)
for ((top, right, bottom, left), name) in zip(boxes, names):
	y = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		2, (0, 255, 0), 5)
	print(name)
image = cv2.resize(image, (500, 500))
cv2.imshow("Image", image)
cv2.waitKey(0)
