import cv2
import numpy as np 
import os
from statistics import mode

classes = os.listdir('./train')

from keras.models import load_model
model = load_model('model.h5')

mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
prediction = [0]*50


vs = cv2.VideoCapture('./1.mp4')
writer = None
(W, H) = (None, None)
# loop over frames from the video file stream
count = 0
while True:
    (grabbed, frame) = vs.read()
    
    if not grabbed:
        break
    
    (H, W) = frame.shape[:2]
        
    output = frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224)).astype("float32")
    frame -= mean

    preds = model.predict(np.expand_dims(frame, axis=0))[0]

    prediction[np.argmax(np.array(preds))] += 1

    i = np.argmax(np.array(prediction))




    # prediction.append(preds)
    # results = np.array(prediction).mean(axis=0)
    
    
    # i = np.argmax(results)
    # prediction.append(i)
    

    # if count>120:
    #     prediction.pop(0)

#     i = mode(prediction)
#     (_, idx, counts) = np.unique(a, return_index=True, return_counts=True)
# index = idx[np.argmax(counts)]
# mode = a[index]

    label = classes[i]

    
    # i = mode(prediction)
    # label = classes[i]

    text = "activity: {}".format(label)
    cv2.putText(output, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter('help', fourcc, 30, (W, H), True)
    # write the output frame to disk
    writer.write(output)
    # show the output image
    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    count += 1
    
# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
