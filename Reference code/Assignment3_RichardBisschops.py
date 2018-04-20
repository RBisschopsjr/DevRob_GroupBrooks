#Made by Richard Bisschops, s4448545
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
global foundABall #Boolean for when we find a ball to shut off some other functions.
global busyMoving #Boolean for when we are moving and should not make other movements.
global motionProxy 
global antiOw #Boolean for when we got hurt and we need to correct ourselves.
global theEnd #Boolean for when we are ending the program.
PORT=9559
IP="192.168.1.103"
memory = ALProxy("ALMemory", IP, PORT)
tts =naoqi.ALProxy("ALTextToSpeech", IP, PORT)
foundABall = False
busyMoving = False
motionProxy = naoqi.ALProxy("ALMotion", IP, PORT)
antiOw= False
theEnd =False

class ReactToTouch(ALModule):
    #Module to react to touches. Meant to be used as avoiding objects on the way by walking back after hitting them.
    def __init__(self,name):
        try:
            p= ALProxy(name)
            p.exit()
        except:
            pass
        ALModule.__init__(self,name)
        self.tts = ALProxy("ALTextToSpeech")
        memory.subscribeToEvent("TouchChanged", name, "onTouched")

    def onTouched(self,strVarName,value):
        #If we hit anything with any of our touch sensors, and we are not ending
        #the program and we did not discover a ball, we should take a few steps back.
        if (not foundABall and not theEnd):
            antiOw =True
            busyMoving =True
            memory.unsubscribeToEvent("TouchChanged", "richard")
            motionProxy.stopMove()
            tts.say("Ow!")

            motionProxy.moveTo(-0.2, 0, 0) #Take a few steps back.
        
            memory.subscribeToEvent("TouchChanged", name, "onTouched")
            antiOw= False
            busyMoving=False #Free to move again.

if __name__ == "__main__":
    videoProxy = naoqi.ALProxy('ALVideoDevice', IP, PORT)
    #React to all touches.
    pythonBroker = ALBroker("pythonBroker", "0.0.0.0", 9600, IP, PORT)
    richard = ReactToTouch("richard")
    
    cam_name="camera"   #Creates an identifier for the camera subscription
    cam_type = 0        #0 for top camera, 1 for bottom camera
    res =1              # 320x240
    colspace=13         #BGR colorspace
    fps=10              #The requested frames per second

    #Horizontal head joint movements.
    headJointsHori = "HeadYaw"
    anglesHeadHoriLeft = [-0.1]
    anglesHeadHoriRight =[0.1]
    #Vertical head joint movements.
    headJointsVerti = "HeadPitch"
    anglesHeadVertiUp = [-0.1]
    anglesHeadVertiDown = [0.1]

    #General head joints movements
    timesHead = [0.1]
    isAbsolute =False

    cams = videoProxy.getSubscribers()
    for cam in cams:
        videoProxy.unsubscribe(cam)

    #subscribing a camera returns a string identifier to be used later on.
    cam = videoProxy.subscribeCamera(cam_name,cam_type,res,colspace,fps)

    randomMax = 3.14
    randomMin = -3.14

    i=0
    motionProxy.setStiffnesses(headJointsHori, 0.8) #Set stiffness of limbs.
    motionProxy.setStiffnesses(headJointsVerti,0.8)
    postureProxy = naoqi.ALProxy("ALRobotPosture", IP, PORT)
    postureProxy.goToPosture("Stand", 0.8)
    sidewaysSpeed = 0
    speed = 0.5
    timeSinceStartMovement = time.time()
    tts.say("Hello. I will start my epic quest for the blue ball. Enjoy.")
    while True:
        
        #image_container contains iinfo about the image
        image_container = videoProxy.getImageRemote(cam)
        
        #get image width and height
        width=image_container[0]
        height=image_container[1]
        
        #the 6th element contains the pixel data
        values = map(ord,list(image_container[6]))

        image=np.array(values, np.uint8).reshape((height, width,3))
        
        cv2.imwrite("ballimage.png", image)
        image=cv2.imread("ballimage.png")
        
        lower_blue = np.array([70,50,50], dtype = np.uint8)
        upper_blue = np.array([170,255,255], dtype=np.uint8)
        
        #convert to a hsv colorspace
        hsvImage=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        
        #Create a treshold mask
        color_mask=cv2.inRange(hsvImage,lower_blue,upper_blue)
        
        #apply the mask on the image
        blue_image = cv2.bitwise_and(image,image,mask=color_mask)
        
        kernel=np.ones((9,9),np.uint8)
        #Remove small objects
        opening =cv2.morphologyEx(color_mask,cv2.MORPH_OPEN,kernel)
        #Close small openings
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        #Apply a blur to smooth the edges
        smoothed_mask = cv2.GaussianBlur(closing, (9,9),0)
        
        #Apply our (smoothend and denoised) mask
        #to our original image to get everything that is blue.
        blue_image = cv2.bitwise_and(image,image,mask=smoothed_mask)
        
        #Get the grayscale image (last channel of the HSV image
        gray_image = blue_image[:,:,2]
        
        #Use a hough transform to find circular objects in the image.
        circles = cv2.HoughCircles(
            gray_image,             #Input image to perform the transformation on
            cv2.HOUGH_GRADIENT,     #Method of detection
            1,                      #Ignore this one
            5,                      #Min pixel dist between centers of detected circles
            param1=200,             #Ignore this one as well
            param2=20,              #Accumulator threshold: smaller = the more (false) circles
            minRadius=5,            #Minimum circle radius
            maxRadius=100)          #Maximum circle radius
        
        #Get the first circle if any. If no circle is available, either stop
        #with looking for blue ball to start finding a new one, or keep searching.
        try:
            circle = circles[0,:][0]
            if not antiOw:
                motionProxy.stopMove()
            busyMoving = False
            if not foundABall:
                foundABall = True
                tts.say("Found the blue ball! Focusing on it")
            
            
            #Draw the detected circle on the original image
            cv2.circle(image, (circle[0], circle[1]), circle[2], (0,255,0),2)
            #Try to get the ball in the area of 145-155 145-155.
            if not antiOw:
                if circle[0]>155:
                    motionProxy.angleInterpolation(headJointsHori, anglesHeadHoriLeft, timesHead, isAbsolute)
                elif circle[0]<145:
                    motionProxy.angleInterpolation(headJointsHori, anglesHeadHoriRight, timesHead, isAbsolute)
                if circle[1]>155:
                    motionProxy.angleInterpolation(headJointsVerti, anglesHeadVertiDown, timesHead, isAbsolute)
                elif circle[1]<145:
                    motionProxy.angleInterpolation(headJointsVerti, anglesHeadVertiUp, timesHead, isAbsolute)
        except:
            if foundABall: #Lost blue ball, reposition head.
                motionProxy.angleInterpolation(headJointsHori, [0], [1.0], True)
                motionProxy.angleInterpolation(headJointsVerti, [0], [1.0], True)
                foundABall= False
                tts.say("I lost the blue ball. Where did it go?")
            elif not antiOw: #Searching for blue ball if we are not avoiding.
                vertiRand = random.uniform(-0.5,0.5)
                horiRand = random.uniform(-0.5,0.5)
                motionProxy.angleInterpolation(headJointsHori, horiRand, [0.5], isAbsolute)
                motionProxy.angleInterpolation(headJointsVerti, vertiRand, [0.5], isAbsolute)
        #Show the image with the ball if present.
        cv2.imshow("Detected image",image)


        #Stop walking with random direction.
        if (time.time()-timeSinceStartMovement >= 3 and not antiOw):
            motionProxy.stopMove()
            busyMoving = False

        #Walk into a new random direction for three seconds if not interrupted.
        if not busyMoving and not foundABall:
            rotation = random.uniform(randomMin,randomMax)
            motionProxy.move(speed, sidewaysSpeed, rotation)
            busyMoving=True
            timeSinceStartMovement =time.time()

        #Reset touch sensors to feel something new.
        try:
            richard = ReactToTouch("richard")
        except:
            pass

        #Pressing exit key will stop the program. Placing robot back into proper position.
        if cv2.waitKey(33) ==27:
            theEnd=True
            busyMoving=True
            videoProxy.unsubscribe(cam)
            motionProxy.stopMove()
            tts.say("End of the demonstration.")
            postureProxy.goToPosture("Sit", 0.5)
            motionProxy.rest()
            pythonBroker.shutdown()
            break #break the while loop
