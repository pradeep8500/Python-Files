import cv2
import os
import shutil
basedir = 'faces'
def rename_folders(face_path,groups_path):
	for fn in os.listdir(face_path):
		
		if fn.startswith('####'):
			try:
				image=(os.listdir(face_path+'/'+fn)[0])
				image = cv2.imread(face_path+'/'+fn+'/'+image)
				cv2.imshow("title", image)
				cv2.waitKey(10)
				a=input('enter name of folder---:')
				os.rename(face_path+'/'+str(fn), basedir+'/'+str(a))	
				os.rename(groups_path+'/'+str(fn), groups_path+'/'+str(a))

			except OSError as err:
				for f in os.listdir(face_path+'/'+fn):
					shutil.move(face_path+'/'+fn+'/'+f,face_path+'/'+a)
				os.rmdir(face_path+'/'+fn)	
				for f in os.listdir(groups_path+'/'+fn):
					shutil.move(groups_path+'/'+fn+'/'+f,groups_path+'/'+a)
				os.rmdir(groups_path+'/'+fn)
		                        
rename_folders('faces','group1')
