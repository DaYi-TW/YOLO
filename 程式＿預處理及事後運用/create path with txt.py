import os
from random import sample
import numpy as np
folder="/Users/dayi/darknet/retrain/image"
filepath=os.listdir(folder)
folder="./retrain/image"
txt=[]
jpg=[]
for filename in filepath:
    if filename.find(".txt") != -1:
        txt.append(folder+"/"+filename)
    elif filename.find(".jpg") != -1:
        jpg.append(folder+"/"+filename)

sampletxt=sample(jpg,len(jpg))
train_jpg=sampletxt[:int(len(jpg)*0.8)]
test_jpg=sampletxt[(int(len(jpg)*0.8)+1):]

np.savetxt("/Users/dayi/darknet/retrain/train.txt", train_jpg, fmt='%s')
np.savetxt("/Users/dayi/darknet/retrain/test.txt", test_jpg, fmt='%s')

