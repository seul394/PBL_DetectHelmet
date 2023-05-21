import numpy as np
import cv2
import time
import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from uuid import uuid4 #new
import os
import pyrebase #new

cred = credentials.Certificate('pbl-safety-management-system-firebase-adminsdk-a15rf-cec8cb6e9e.json')

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://pbl-safety-management-system-default-rtdb.firebaseio.com/',
    'storageBucket': 'pbl-safety-management-system.appspot.com'
})

dir = db.reference('cameraSensor')

box_time = 0
frame = None
min_confidence = 0.5
width = 288
height = 0
show_ratio = 1.0
title_name = 'Custom Yolo'
# Load Yolo
net = cv2.dnn.readNet("./backup_tiny/yolov3-tiny_final.weights", "./yolov3-tiny.cfg")
classes = []
with open("./classes.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
color_lists = np.random.uniform(0, 255, size=(len(classes), 3))

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#output_layers = []
#for i in net.getUnconnectedOutLayers():
#    output_layers.append(layer_names[i - 1])

#camera capture
def capture():
    captureTime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = captureTime+'.jpg'
    imageDir = db.reference('image/'+captureTime)
    print(filename)
    #storage update
    bucket = storage.bucket()
    cv2.imwrite('capture_image/'+filename, frame)
    blob = bucket.blob('capture_image/'+filename)
    new_token = uuid4() #new
    metadata = {"firebaseStorageDownloadTokens": new_token} #new
    blob.metadata = metadata #new
    blob.upload_from_filename('capture_image/'+filename)
    
    #realtime database update
    date = datetime.datetime.now().strftime("%Y%m%d")
    #imageDir.update({'imageUrl': blob.public_url})
    imageDir.update({'fileName': filename}) #new
    imageDir.update({'location': "공사장1"})
    imageDir.update({'time': date})

#delete local file
def clearAll():
    path = 'capture_image'
    os.system('rm -rf %s/*' % path)
    print('delete captures')

def detectAndDisplay(image):    
    h, w = image.shape[:2]
    height = int(h * width / w)
    img = cv2.resize(image, (width, height))

    #blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
    #blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), swapRB=True, crop=False)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (0, 0), swapRB=True, crop=False)
    #blob = cv2.dnn.blobFromImage(img, 1.0, (0, 0), swapRB=True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)
    
    confidences = []
    names = []
    boxes = []
    colors = []


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > min_confidence:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                names.append(classes[class_id])
                colors.append(color_lists[class_id])
                
    #10초 동안 no_helmet이 없으면 box_time 초기화
    global box_time
    now = time.time()
    if (now - box_time) > 10:   
        box_time = 0
        print('box_time 변경')
        dir.update({'new': '0'})
        clearAll()
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        print(box_time)
        if i in indexes:
            x, y, w, h = boxes[i]
            label = '{} {:,.2%}'.format(names[i], confidences[i])
            color = colors[i]
            print(i, label, x, y, w, h)
            
            #firebase
            if (names[i] == 'Helmet'):
                dir.update({'helmet':'0'})
            else:
                dir.update({'helmet':'1'})
                if box_time == 0:   # 새로운 no_helmet 발견 시 new를 1로 변경
                    dir.update({'new': '1'})
                    capture()
                box_time = time.time()
                

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), font, 1, color, 2)
    cv2.imshow(title_name, img)

vs = cv2.VideoCapture(0)
time.sleep(2.0)
if not vs.isOpened:
    print('### Error opening video ###')
    exit(0)
while True:
    ret, frame = vs.read()
    if frame is None:
        print('### No more frame ###')
        vs.release()
        break
    detectAndDisplay(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        dir.update({'new': '0'})
        break

vs.release()
cv2.destroyAllWindows()
