# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png 
# python3 recognize_faces_image.py --encodings faces.pickle --image examples/p1.jpg 
# python recognize_faces_image.py --encodings faces.pickle --image qa.jpg


# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2
import os
import shutil

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# load the input image and convert it from BGR to RGB
path=args["image"]
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect the (x, y)-coordinates of the bounding boxes corresponding
# to each face in the input image, then compute the facial embeddings
# for each face

print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb,
	model=args["detection_method"])
print('boxes are formed',boxes)
encodings = face_recognition.face_encodings(rgb, boxes)
# initialize the list of names for each face detected
names = []
# loop over the facial embeddings
for encoding in encodings:
	# attempt to match each face in the input image to our known
	# encodings
	#print('lllllllllll',face_recognition.face_distance(data["encodings"],encoding))
	matches = face_recognition.compare_faces(data["encodings"],encoding,tolerance=0.42)
	name = "Unknown"
	#print(matches)

	# check to see if we have found a match
	if True in matches:
		# find the indexes of all matched faces then initialize a
		# dictionary to count the total number of times each face
		# was matched
		matchedIdxs = [i for (i, b) in enumerate(matches) if b]
		counts = {}
		#print(data["names"])
		# loop over the matched indexes and maintain a count for
		# each recognized face face
		for i in matchedIdxs:
			name = data["names"][i]
			counts[name] = counts.get(name, 0) + 1
		print(counts)
		# determine the recognized face with the largest number of
		# votes (note: in the event of an unlikely tie Python will
		# select first entry in the dictionary)
		name = max(counts, key=counts.get)
	        #saved_path='faces/'+name
	        #shutil.copy(path,saved_path)
	# update the list of names
	names.append(name)

# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
	# draw the predicted face name on the image
	print('jhhhhhhhhhhhhhhhhhhhhhhhh',top, right, bottom, left)
	cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
	y = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		2, (0, 255, 0), 4)
	print(name)
image = cv2.resize(image, (500, 500))
# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
