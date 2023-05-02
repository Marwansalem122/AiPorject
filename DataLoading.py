import numpy as np 
import cv2 
import os 
class SimpleDatasetLoader:
	
	# defines the constructor to SimpleDatasetLoader  > 
	# where we can optionally pass in a list of image preprocessors >
	# (e.g., SimplePreprocessor) that can be sequentially applied to a given input image.
	def __init__(self, preprocessors=None):
		# store the image preprocessor like simplePreprocessor we made 
		self.preprocessors = preprocessors
		# if the preprocessors are None, initialize them as an
		# empty list
		if self.preprocessors is None:
			self.preprocessors = []
			
	def load(self, imagePaths, verbose=-1):
		# initialize the list of features and labels
		data = []
		labels = []
		# loop over the input images
		for (i, imagePath) in enumerate(imagePaths):
			# load the image and extract the class label assuming
			# that our path has the following format:
			# /path/to/dataset/{class}/{image}.jpg
			image = cv2.imread(str(r""+imagePath))
			label = imagePath.split(os.path.sep)[-2]
			#print("this is before :"+label)			
			
			#print("this is after :"+l)

			#print(label)
			# check to see if our preprocessors are not None
			if self.preprocessors is not None:
				# loop over the preprocessors and apply each to
				# the image
				for p in self.preprocessors:
					image = p.preprocess(image)
			# treat our processed image as a "feature vector"
			# by updating the data list followed by the labels
			data.append(image)
			labels.append(label)
		return (np.array(data), np.array(labels))
	
