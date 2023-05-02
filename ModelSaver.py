import pickle
def SaveModel(model):
    knnPickle = open('knnpickle_file', 'wb') 
    pickle.dump(model, knnPickle)  
    knnPickle.close()