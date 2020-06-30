import sys
import os
import dlib
import glob
import cv2
import shutil
import datetime
from imutils import paths
import face_recognition
import argparse
import pickle
import json

def cluster_images(faces_folder_path,output_folder_path,output_folder_path1,predictor_path,face_rec_model_path,output_folder_path2,json_name):
	# Load all the models we need: a detector to find the faces, a shape predictor
	# to find face landmarks so we can precisely localize the face, and finally the
	# face recognition model.
	from_date = (datetime.datetime.now())
	detector = dlib.get_frontal_face_detector()
	sp = dlib.shape_predictor(predictor_path)
	facerec = dlib.face_recognition_model_v1(face_rec_model_path)
	if os.path.exists(json_name):
		img_dict = json.loads(open(json_name, "r").read())
	else:
		img_dict={}
	descriptors = []
	images = []
	image_names=[]
	# Now find all the faces and compute 128D face descriptors for each face.
	for l,f in enumerate(glob.glob(os.path.join(faces_folder_path, "*"))):
		print("Processing file: {}".format(f))
		print("[INFO] processing image {}/{}".format(l + 1,
			len(os.listdir(faces_folder_path))))
		img = dlib.load_rgb_image(f)
		img_name=f.split(os.path.sep)[-1]
		#print('img_name',img_name)		
		# Ask the detector to find the bounding boxes of each face. The 1 in the
		# second argument indicates that we should upsample the image 1 time. This
		# will make everything bigger and allow us to detect more faces.
		dets = detector(img)
		print("Number of faces detected: {}".format(len(dets)))

		# Now process each face we found.
		for k, d in enumerate(dets):

			#d= dlib.rectangle(d.left()-500, d.top()-500,d.right()+50, d.bottom()+50)
			# Get the landmarks/parts for the face in box d.
			shape = sp(img, d)
			# Compute the 128D vector that describes the face in img identified by
			# shape.  
			face_descriptor = facerec.compute_face_descriptor(img, shape)
			descriptors.append(face_descriptor)
			images.append((img, shape))
			image_names.append(img_name)
	# Now let's cluster the faces.  
	labels = dlib.chinese_whispers_clustering(descriptors, 0.4)
	num_classes = len(set(labels))
	for j in range(0, num_classes):
		indices = []
		for i, label in enumerate(labels):
			if label == j:
				indices.append(i)
		if len(indices) >1 :
		
			# Ensure output directory exists
			if not os.path.isdir(output_folder_path+'/####'+str(j)):
				os.makedirs(output_folder_path+'/####'+str(j))

			if not os.path.isdir(output_folder_path1+'/####'+str(j)):
				os.makedirs(output_folder_path1+'/####'+str(j))
			if not os.path.isdir(output_folder_path2):
				os.makedirs(output_folder_path2)
			name_of_images=[]
			# Save the extracted faces
			for i, index in enumerate(indices):
				name_image=currentDT.strftime("%Y-%m-%d %H:%M:%S")
				img, shape = images[index]
				name_of_images.append(image_names[index])
				file_path = os.path.join(output_folder_path+'/####'+str(j),str(name_image)+str(j)+'-'+str(i))
				# The size and padding arguments are optional with default size=150x150 and padding=0.25
				dlib.save_face_chip(img, shape, file_path, size=150, padding=0.25)
				file_path1 = os.path.join(output_folder_path1+'/####'+str(j), str(name_image)+str(j)+'-'+str(i)+'.jpg')
				#image path must end with one of [.bmp, .png, .dng, .jpg, .jpeg] for dlib.save_image
				dlib.save_image(img,file_path1)
				f1 = open(output_folder_path2+'/####'+str(j)+'.txt', "a")
				f1.write(str(name_image)+str(j)+'-'+str(i)+'.jpg'+'\n')
				f1.close()
			#img_dict.update({'####'+str(j): name_of_images} )
			img_dict['####'+str(j)] = name_of_images
	f = open(json_name, "w")
	f.write(json.dumps(img_dict))
	f.close()
	to_date = (datetime.datetime.now())
	print('clustering inages time',to_date-from_date)


def compare_folders(pickle_name,output_folder_path,output_folder_path1,output_folder_path2,json_name):
	data = pickle.loads(open(pickle_name, "rb").read())
	img_dict = json.loads(open(json_name, "r").read())
	for fn in os.listdir(output_folder_path):
		if fn.startswith('####'):
			image=(os.listdir(output_folder_path+'/'+fn)[0])
			image = cv2.imread(output_folder_path+'/'+fn+'/'+image)

			rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			boxes = face_recognition.face_locations(rgb,
				model='hog')
			encodings = face_recognition.face_encodings(rgb, boxes)
			names = []
			for (j,encoding) in enumerate(encodings):
				matches = face_recognition.compare_faces(data["encodings"],
					encoding,tolerance=0.4)
				# check to see if we have found a match
				if True in matches:
					matchedIdxs = [i for (i, b) in enumerate(matches) if b]
					counts = {}
					for i in matchedIdxs:
						name = data["names"][i]
						counts[name] = counts.get(name, 0) + 1
					name = max(counts, key=counts.get)
					for f in os.listdir(output_folder_path+'/'+fn):
						shutil.move(output_folder_path+'/'+fn+'/'+f, output_folder_path+'/'+name)
					for f in os.listdir(output_folder_path1+'/'+fn):
						shutil.move(output_folder_path1+'/'+fn+'/'+f, output_folder_path1+'/'+name)
					with open(output_folder_path2+'/'+name+'.txt', 'a') as outfile:
						with open(output_folder_path2+'/'+fn+'.txt') as infile:
							outfile.write(infile.read())
					list1=img_dict[str(fn)]
					for i in list1:
						img_dict[str(name)].append(str(i))
					img_dict.pop(str(fn),None)
					os.rmdir(output_folder_path+'/'+fn)
					os.rmdir(output_folder_path1+'/'+fn)
					os.remove(output_folder_path2+'/'+fn+'.txt')
	f = open(json_name, "w")
	f.write(json.dumps(img_dict))
	f.close()

def rename_folders(face_path,groups_path,file_path,json_name):
	for fn in os.listdir(face_path):
		img_dict = json.loads(open(json_name, "r").read())
		if fn.startswith('####'):
			try:
				image=(os.listdir(face_path+'/'+fn)[0])
				image = cv2.imread(face_path+'/'+fn+'/'+image)
				image = cv2.resize(image, (500, 500))
				cv2.imshow("title", image)
				cv2.waitKey(500)
				a=input('enter name of folder---:')
				os.rename(face_path+'/'+fn, face_path+'/'+a)	
				os.rename(groups_path+'/'+fn, groups_path+'/'+a)
				os.rename(file_path+'/'+fn+'.txt', file_path+'/'+a+'.txt')
				img_dict[str(a)] = img_dict.pop(str(fn))

			except OSError as err:
				for f in os.listdir(face_path+'/'+fn):
					shutil.move(face_path+'/'+fn+'/'+f,face_path+'/'+a)
				os.rmdir(face_path+'/'+fn)	
				for f in os.listdir(groups_path+'/'+fn):
					shutil.move(groups_path+'/'+fn+'/'+f,groups_path+'/'+a)
				os.rmdir(groups_path+'/'+fn)
				with open(file_path+'/'+a+'.txt', 'a') as outfile:
					with open(file_path+'/'+fn+'.txt') as infile:
						outfile.write(infile.read())
				if os.path.exists(file_path+'/'+fn+'.txt'):
					os.remove(file_path+'/'+fn+'.txt')
				list1=img_dict[str(fn)]
				for i in list1:
					img_dict[str(a)].append(str(i))
				img_dict.pop(str(fn),None)

		f = open(json_name, "w")
		f.write(json.dumps(img_dict))
		f.close()

def training_images(output_folder_path,pickle_name):
	from_date = (datetime.datetime.now())
       
        

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

		# load the input image and convert it from RGB (OpenCV ordering)
		# to dlib ordering (RGB)
		image = cv2.imread(imagePath)
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# detect the (x, y)-coordinates of the bounding boxes
		# corresponding to each face in the input image
		boxes = face_recognition.face_locations(rgb,
			model='hog')

		# compute the facial embedding for the face
		encodings = face_recognition.face_encodings(rgb, boxes)

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
	to_date = (datetime.datetime.now())
	print('training inages time',to_date-from_date)

predictor_path = 'shape_predictor_5_face_landmarks.dat'
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
faces_folder_path = '1mb'
output_folder_path = 'faces'
output_folder_path1 = 'groups1'
output_folder_path2 = 'files'
pickle_name='faces1.pickle'
json_name='images.json'

currentDT = datetime.datetime.now()

cluster_images(faces_folder_path,output_folder_path,output_folder_path1,predictor_path,face_rec_model_path,output_folder_path2,json_name)
if os.path.exists(pickle_name): 
	compare_folders(pickle_name,output_folder_path,output_folder_path1,output_folder_path2,json_name)
rename_folders(output_folder_path,output_folder_path1,output_folder_path2,json_name)	
training_images(output_folder_path,pickle_name)    
