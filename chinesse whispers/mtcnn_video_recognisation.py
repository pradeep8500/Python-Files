from mtcnn.mtcnn import MTCNN
import face_recognition
import cv2
import imutils
import pickle

stream = cv2.VideoCapture('videos/VID_20191213_162943.mp4')
writer = None
data = pickle.loads(open('faces11.pickle', "rb").read())
while True:
		(grabbed, frame) = stream.read()
		if not grabbed:
			break
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		rgb = imutils.resize(frame, width=750)
		r = frame.shape[1] / float(rgb.shape[1])
		detector = MTCNN()
		faces = detector.detect_faces(frame)
		boxes=[]
		for result in faces:
			x, y, width, height = result['box']
			boxes.append((y,x+width,y+height,x))
			cv2.rectangle(frame,(x,y),(x+width,y+height),(0,255,0),2)
		encodings = face_recognition.face_encodings(rgb, boxes)
		names = []

		# loop over the facial embeddings
		if encodings:
			for encoding in encodings:
				# attempt to match each face in the input image to our known
				# encodings
				matches = face_recognition.compare_faces(data["encodings"],
	encoding,tolerance=0.4)
				name = "Unknown"

				# check to see if we have found a match
				if True in matches:
					# find the indexes of all matched faces then initialize a
					# dictionary to count the total number of times each face
					# was matched
					matchedIdxs = [i for (i, b) in enumerate(matches) if b]
					counts = {}

					# loop over the matched indexes and maintain a count for
					# each recognized face face
					for i in matchedIdxs:
						name = data["names"][i]
						counts[name] = counts.get(name, 0) + 1

					# determine the recognized face with the largest number
					# of votes (note: in the event of an unlikely tie Python
					# will select first entry in the dictionary)
					name = max(counts, key=counts.get)
				
				# update the list of names
				names.append(name)
				print('names',names)
		for ((top, right, bottom, left), name) in zip(boxes, names):
			
			y = top - 15 if top - 15 > 15 else top + 15
			cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
				0.75, (0, 255, 0), 2)


		if writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter('output/VID_20191213_162943.avi', fourcc, 24,
				(frame.shape[1], frame.shape[0]), True)
		if writer is not None:
			writer.write(frame)
stream.release()
