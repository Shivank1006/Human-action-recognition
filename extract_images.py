import os 
import cv2
import math


# temp = 'train'
temp = 'test'

# list all the folder categories
folders = os.listdir('./'+temp+'/')
os.mkdir('./'+temp+'_images')

# make folder to save images
for i in folders:
    os.mkdir('./'+temp+'_images/' + i)

# for each folder
#   for each file
#       extract images


for folder in folders:
    files = os.listdir('./'+temp+'/' + folder)
    for file in files:
        count = 0
        videoFile = file
        cap = cv2.VideoCapture('./'+temp+'/'+folder+'/' + file)   # capturing the video from the given path
        frameRate = cap.get(5) #frame rate
        while(cap.isOpened()):
            frameId = cap.get(1) #current frame number
            ret, frame = cap.read()
            if (frameId>frameRate*4 or ret != True):
                break
            if (frameId % math.floor(frameRate/2) == 0):
            # storing the frames in a new folder named train_1
                filename ='./'+temp+'_images/'+ folder + '/' + file +"_frame%d.jpg" % count
                count+=1
                cv2.imwrite(filename, frame)
        cap.release()
    print('done ' + folder)
