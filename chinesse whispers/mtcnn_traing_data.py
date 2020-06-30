import os
import dlib
from imutils import paths
import pickle
from mtcnn.mtcnn import MTCNN
import face_recognition
import cv2
from matplotlib import pyplot

def training_images(output_folder_path,pickle_name):

	imagePaths = list(paths.list_images(output_folder_path))

	# initialize the list of known encodings and known names
	knownEncodings = []
	knownNames = []

	# loop over the image paths
	for (i, imagePath) in enumerate(imagePaths):
		# extract the person name from the image path
		print("[INFO] processing image {}/{}".format(i + 1,
			len(imagePaths)))
		name = imagePath.split(os.path.sep)[-2]
		#image = cv2.imread(imagePath)
		image = pyplot.imread(imagePath)  ####pyplot gives more images compare to cv2.read
		#rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


		detector = MTCNN()
		faces = detector.detect_faces(image)
		boxes=[]

		for result in faces:
			x, y, width, height = result['box']
			boxes.append((y,x+width,y+height,x))


		encodings = face_recognition.face_encodings(image, boxes)

		# loop over the encodings
		for encoding in encodings:
			# add each encoding + name to our set of known names and
			# encodings
			knownEncodings.append(encoding)
			knownNames.append(name)


	# dump the facial encodings + names to disk
	print("[INFO] serializing encodings...")
	data = {"encodings": knownEncodings, "names": knownNames}
	f = open(pickle_name, "wb")
	f.write(pickle.dumps(data))
	f.close()


output_folder_path = 'hello_faces'
pickle_name='hello.pickle'
training_images(output_folder_path,pickle_name)    
