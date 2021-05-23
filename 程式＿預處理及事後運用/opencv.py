import cv2 as cv
import sys
import matplotlib.pyplot as plt
import matplotlib.image as img
import time
from datetime import datetime
import numpy as np


cap = cv.VideoCapture(0) #打開攝影機
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_eye.xml')
while(True):

    logdir = '/Users/dayi/PycharmProjects/yolo img/' + format(datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))
    #time.sleep(1)
    ret, img = cap.read() #讀取影像
    img=cv.flip(img, 1, dst=None)

    dim_shape = img.shape[:2]

    # 按下 q 鍵離開迴圈
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
    if faces != ():
        a = faces[:,2].tolist()
        index = a.index(max(a))
        faces = faces[index]
        (x, y, w, h) = faces
        #cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #crop_img = img[y:y + h, x:x + w]
        px = str(((x + w/2))*1.0 / dim_shape[1])
        py = str(((y + h/2))*1.0 / dim_shape[0])
        pw = str(w*1.0 / dim_shape[1])
        ph = str(h*1.0 / dim_shape[0])

        #eyes = eye_cascade.detectMultiScale(crop_img, scaleFactor=1.02, minNeighbors=3,minSize=(40,40))

        location ="1" + " " + px + " " + py + " " + pw + " " + ph
        f = open(logdir+'.txt', 'w')
        f.writelines(location)
        f.close()
        cv.imwrite(logdir+'.jpg', img)

    cv.imshow('img', img)
    if cv.waitKey(1) == ord('q'):
        cap.release()
        cv.destroyAllWindows()
        break
