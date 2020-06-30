# face detection with mtcnn on a photograph
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
import cv2
 
# draw an image with detected objects
def draw_image_with_boxes(filename, result_list):
	# load the image
	data = pyplot.imread(filename)
	# plot the image
	pyplot.imshow(data)
	# get the context for drawing boxes
	ax = pyplot.gca()
	# plot each box
	for result in result_list:
		# get coordinates
		x, y, width, height = result['box']
		# create the shape
		rect = Rectangle((x, y), width, height, fill=False, color='red')


		#image1 = cv2.imread(filename)
		#cv2.rectangle(image1, (2130, 1403), (1745, 1018), (0, 255, 0), 4)
		#cv2.rectangle(image1,(x,y),(x+width,y+height),(0,255,0),2)
		#image1 = cv2.resize(image1, (500, 500))
		# show the output image
		#cv2.imshow("Image", image1)
		#cv2.waitKey(0)
	
		# draw the box
		ax.add_patch(rect)
	# show the plot
	pyplot.show()
 
filename = '1574851494379.JPEG'
# load image from file
pixels = pyplot.imread(filename)
# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
faces = detector.detect_faces(pixels)
print('faces',faces)
# display faces on the original image
draw_image_with_boxes(filename, faces)
