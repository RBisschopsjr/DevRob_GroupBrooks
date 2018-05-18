import cv2
import numpy as np
import naoqi
import time
import random
from naoqi import ALModule
from naoqi import ALProxy
from naoqi import ALBroker


global IP
global PORT
global memory
global tts
global motionProxy
global headJointsHori
global headJointsVerti
global videoProxy
global pythonBroker

PORT=9559
IP="192.168.1.143"
memory = ALProxy("ALMemory", IP, PORT)
tts =naoqi.ALProxy("ALTextToSpeech", IP, PORT)
motionProxy = naoqi.ALProxy("ALMotion", IP, PORT)
headJointsHori = "HeadYaw"
headJointsVerti = "HeadPitch"
videoProxy = naoqi.ALProxy('ALVideoDevice', IP, PORT)
pythonBroker = ALBroker("pythonBroker", "0.0.0.0", 9600, IP, PORT)


cam_name = "camera"  # Creates an identifier for the camera subscription
cam_type = 0  # 0 for top camera, 1 for bottom camera
res = 1  # 320x240
colspace = 13  # BGR colorspace
fps = 10  # The requested frames per second

cams = videoProxy.getSubscribers()
for cam in cams:
    videoProxy.unsubscribe(cam)

cam = videoProxy.subscribeCamera(cam_name, cam_type, res, colspace, fps)

# image_container contains info about the image
image_container = videoProxy.getImageRemote(cam)

# get image width and height
width = image_container[0]
height = image_container[1]

# the 6th element contains the pixel data
values = map(ord, list(image_container[6]))

image = np.array(values, np.uint8).reshape((height, width, 3))

cv2.imwrite("faceimage.png", image)
image = cv2.imread("faceimage.png")

face_cascade = cv2.CascadeClassifier('C:\Users\gebruiker\Documents\Master\Developmental Robotics\opencv\data\haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\Users\gebruiker\Documents\Master\Developmental Robotics\opencv\data\haarcascades\haarcascade_eye.xml')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
if len(faces) > 0 :
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    tts.say("I found a face")
else:
    tts.say("I did not find a face")
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()