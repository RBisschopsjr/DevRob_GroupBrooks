import cv2
import numpy as np
import naoqi
import time
import random
from naoqi import ALModule
from naoqi import ALProxy
from naoqi import ALBroker

PORT=9559
IP="192.168.1.103"
headJointsHori = "HeadYaw"
headJointsVerti = "HeadPitch"
isAbsolute=True
motionProxy = naoqi.ALProxy("ALMotion", IP, PORT)
postureProxy = naoqi.ALProxy("ALRobotPosture", IP, PORT)
motionProxy.setStiffnesses(headJointsHori, 0.8) #Set stiffness of limbs.
motionProxy.setStiffnesses(headJointsVerti,0.8)
postureProxy.goToPosture("Sit", 0.5)
try:
    while True:
        vertiRand = random.uniform(-0.5,0.5)
        horiRand = random.uniform(-0.5,0.5)
        motionProxy.angleInterpolation(headJointsHori, horiRand, [0.5], isAbsolute)
        motionProxy.angleInterpolation(headJointsVerti, vertiRand, [0.5], isAbsolute)
except:
    postureProxy.goToPosture("Sit", 0.5)
    motionProxy.rest()
