import cv2
import numpy as np 
import os
import math

temp = 'test'

# list all the folder categories
folders = os.listdir('./'+temp+'/')

from keras.models import load_model
model = load_model('model.h5')

mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
predicted = []
target = []
j = 0

for folder in folders:
    files = os.listdir('./'+temp+'/' + folder)
    # y = [0]*50
    # y[j] = 1
    for file in files:
        prediction = [0]*50
        # y_prime = [0]*50
        videoFile = file
        cap = cv2.VideoCapture('./'+temp+'/'+folder+'/' + file)   # capturing the video from the given path
        frameRate = cap.get(5) #frame rate
        while(cap.isOpened()):
            frameId = cap.get(1) #current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frameId % math.floor(frameRate/2) == 0):
            # storing the frames in a new folder named train_1
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224)).astype("float32")
                frame -= mean
                preds = model.predict(np.expand_dims(frame, axis=0))[0]
                prediction[np.argmax(np.array(preds))] += 1

        cap.release()
        i = np.argmax(np.array(prediction))
        # y_prime[i] = 1
        predicted.append(i)
        target.append(j)
    print('done ' + folder)
    j += 1
    
print(j)
f = open('f1.txt', 'w')
f.write('Predicted= ')
f.write(str(predicted) + '\n')
f.write('Target= ')
f.write(str(target))
f.close()

# print(target)
# release the file pointers
print("[INFO] cleaning up...") 