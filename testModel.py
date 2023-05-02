from PreProcessing import SimplePreprocessor 
from DataLoading import SimpleDatasetLoader
import pickle
import cv2 

p = SimplePreprocessor(32,32)
img = p.preprocess(cv2.imread(
	r"raw-img\spider\e83cb00a2bf0053ed1584d05fb1d4e9fe777ead218ac104497f5c97ca5ecb3b9_640.jpg"))
img = img.reshape(1,3072)

loaded_model = pickle.load(open('knnpickle_file','rb'))

result = loaded_model.predict(img) 
	
names = [ "butterfly",  "cat",  "chicken",  "cow", "dog", "elefant", 
         "horse", "sheep", "spider","squirrel"]

print(names[int(result[0])])
