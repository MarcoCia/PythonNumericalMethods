"""
@author: marcocianciotta
"""
# Import necessary packages
from __future__ import print_function
import os
import sys
import cv2
import numpy as np

# Create data matrix from a list of images
def createDataMatrix(images):
	print("Creazione della matrice dei dati",end=" ... ")
	
	numImages = len(images)
	sz = images[0].shape
	data = np.zeros((numImages, sz[0] * sz[1] * sz[2]), dtype=np.float32)
	for i in range(0, numImages):
		image = images[i].flatten()
		data[i,:] = image
	
	print("DONE")
	return data

# Read images from the directory
def readImages(path):
	print("Lettura delle immagini " + path, end="...")
	# Create array of array of images.
	images = []
	# List all files in the directory and read points from text files one by one
	for filePath in sorted(os.listdir(path)):
		fileExt = os.path.splitext(filePath)[1]
		if fileExt in [".jpg", ".jpeg"]:

			# Add to array of images
			imagePath = os.path.join(path, filePath)
			im = cv2.imread(imagePath)

			if im is None :
				print("image:{} Non letta Correttamente".format(imagePath))
			else :
				# Convert image to floating point
				im = np.float32(im)/255.0
				# Add image to list
				images.append(im)
				# Flip image 
				imFlip = cv2.flip(im, 1);
				# Append flipped image
				images.append(imFlip)
	numImages = int(len(images) / 2)
	# Exit if no image found
	if numImages == 0 :
		print("Nessuna immagine trovata")
		sys.exit(0)

	print(str(numImages) + " files read.")
	return images

# Add the weighted eigen faces to the mean face 
def createNewFace(*args):
	# Start with the mean image
	output = averageFace
	
	# Add the eigen faces with the weights
	for i in range(0, NUM_EIGEN_FACES):
		'''
		OpenCV does not allow slider values to be negative. 
		So we use weight = sliderValue - MAX_SLIDER_VALUE / 2
		''' 
		sliderValues[i] = cv2.getTrackbarPos("Weight" + str(i), "Trackbars");
		weight = sliderValues[i] - MAX_SLIDER_VALUE/2
		output = np.add(output, eiFa[i] * weight)

	# Display Result at 2x size
	output = cv2.resize(output, (0,0), fx=2, fy=2)
	cv2.imshow("Result", output)

def resetSliderValues(*args):
	for i in range(0, NUM_EIGEN_FACES):
		cv2.setTrackbarPos("Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE/2));
	createNewFace()

if __name__ == '__main__':

	# Number of EigenFaces
	NUM_EIGEN_FACES = 10

	# Maximum weight
	MAX_SLIDER_VALUE = 255

	# Directory contenente le immagini
	dirName = "images"

	# Lettura immagini
	images = readImages(dirName)
	
	# Size of images
	sz = images[0].shape

	# Create data matrix for PCA.
	data = createDataMatrix(images)

	# Compute the eigenvectors from the stack of images created
	print("Calculating PCA ", end="...")
	mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=NUM_EIGEN_FACES)
	print ("DONE")

	averageFace = mean.reshape(sz)

	eiFa = []; 

	for eigenVector in eigenVectors:
		eigenFace = eigenVector.reshape(sz)
		eiFa.append(eigenFace)

    
    	#Il codice che segue, serve per la creazione
    	#della finestra per la demo
    
    
   
	# Create window for displaying Mean Face
	cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)
	
	# Display result at 2x size
	output = cv2.resize(averageFace, (0,0), fx=2, fy=2)
	cv2.imshow("Result", output)

	# Create Window for trackbars
	cv2.namedWindow("Trackbars", cv2.WINDOW_AUTOSIZE)

	sliderValues = []
	
	# Create Trackbars
	for i in range(0, NUM_EIGEN_FACES    ):
		sliderValues.append(int(MAX_SLIDER_VALUE/2))
		cv2.createTrackbar( "Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE/2), MAX_SLIDER_VALUE, createNewFace)
	
	# You can reset the sliders by clicking on the mean image.
	cv2.setMouseCallback("Result", resetSliderValues);
	
	print('''Utilizzare la Demo per Osservare come cambia l'immagine al modificare dei pesi''')

	cv2.waitKey(0)
	cv2.destroyAllWindows()
