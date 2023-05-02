# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from pyimagesearch.preprocessing import SimplePreprocessor
#from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths

import numpy as np 
import cv2 
from DataLoading import SimpleDatasetLoader
from PreProcessing import SimplePreprocessor
import PreProcessing


names = [ "butterfly",  "cat",  "chicken",  "cow", "dog", "elefant", 
         "horse", "sheep", "spider","squirrel"]
print(len(names))




imagePaths = list(paths.list_images("raw-img"))
# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
#After loading our images from disk, the data NumPy array has a .shape of (3000, 32, 32, 3), indicating there are 3,000 images in the dataset, each 32×32 pixels with 3 channels.


(data, labels) = sdl.load(imagePaths, verbose=1)



#Note that 
#However, to apply the k-NN algorithm, 
#we need to “flatten” our images from a 3D representation to a single list of pixel 
#intensities. We accomplish this, Line 30 calls the .reshape method on the data NumPy array,
 #flattening the 32×32×3 images into an array with shape (2000, 3072).
  #The actual image data hasn’t changed at all — the images are simply represented as a list of 3,000 entries, each of
  # 3,072-dim (32×32×3 = 3,072).
#verbose represents is the number of images we will print when we use our own method for printing 

data = data.reshape((data.shape[0], 3072))
# show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(
	data.nbytes / (1024 * 1024.0)))


# encode the labels as integers
#convert our labels (represented as strings) to integers where we have one unique 
 #integer per class. This conversion allows us to map the cat class to the integer 0, 
  #the dog class to integer 1, and the panda class to integer 2. Many machine learning algorithms assume that 
   #the class labels are encoded as integers, so it’s important that we get in the habit of performing this step.
le = LabelEncoder()

labels = le.fit_transform(labels)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=1)
#It is common to use the variable X to refer to a dataset that contains the 
 #data points we’ll use for training and testing while y refers to the class labels
  #Therefore, we use the variables trainX and testX to refer to the training and testing examples,
   #respectively. The variables trainY and testY are our training and testing labels


# train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=10,
	n_jobs=4)
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX),
	target_names=names))



import ModelSaver
ModelSaver.SaveModel(model)