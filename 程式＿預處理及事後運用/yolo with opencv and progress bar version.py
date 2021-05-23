import time
import cv2
import numpy as np
from tqdm import tqdm, trange
from moviepy.editor import VideoFileClip

INPUT_FILE = "/Users/dayi/darknet/data/2019.9.12 日本東京澀谷十字路口.mp4"
OUTPUT_FILE = '/Users/dayi/2019.9.12 日本東京澀谷十字路口.avi'
LABELS_FILE = '/Users/dayi/darknet/data/coco.names'
CONFIG_FILE = '/Users/dayi/darknet/cfg/yolov4.cfg'
WEIGHTS_FILE = '/Users/dayi/darknet/yolov4.weights'
CONFIDENCE_THRESHOLD = 0.1
LABELS = open(LABELS_FILE).read().strip().split("\n")

np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


cap = cv2.VideoCapture(INPUT_FILE)  # 打開攝影機
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(OUTPUT_FILE,fourcc, 24.0, (1280, 720))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for i in trange(total):
    (grabbed, image) = cap.read()  # 讀取影像

    #image = cv2.flip(image, 1, dst=None) #水平翻轉
    (H, W) = image.shape[:2]
    #image = cv2.GaussianBlur(image, (0, 0), 5) #高斯模糊
    image = cv2.addWeighted(image, 1.5, image, -0.5, 0) #銳化
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    #print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    """
    預測框
    """
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE_THRESHOLD:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
                        CONFIDENCE_THRESHOLD)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

    # show the output image
    #cv2.imshow('img', image)
    writer.write(image)


    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

