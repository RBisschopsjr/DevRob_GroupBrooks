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
IP="192.168.1.103"
memory = ALProxy("ALMemory", IP, PORT)
tts =naoqi.ALProxy("ALTextToSpeech", IP, PORT)
motionProxy = naoqi.ALProxy("ALMotion", IP, PORT)
headJointsHori = "HeadYaw"
headJointsVerti = "HeadPitch"
videoProxy = naoqi.ALProxy('ALVideoDevice', IP, PORT)
pythonBroker = ALBroker("pythonBroker", "0.0.0.0", 9600, IP, PORT)

##for c number of trials:
##	while not face found:
##		findFace()
##	random= takeRandomNumberOnPreference
##	PerformBehaviour(random)
##	state if ball was found
##shut down

#TODO: Implement face detection.
def faceFound():
    return randint(0,1)==1

def findFace():
    while not faceFound():
        vertiRand = random.uniform(-0.5,0.5)
        horiRand = random.uniform(-0.5,0.5)
        motionProxy.angleInterpolation(headJointsHori, horiRand, [0.5], isAbsolute)
        motionProxy.angleInterpolation(headJointsVerti, vertiRand, [0.5], isAbsolute)

#TODO: Implement determining what behaviour to pick.
def getChoice():
    return randint(0,1)==1

#TODO: Implement finding object through gaze.
def faceGaze():
    tts.say("I would now perform face gaze if it was implemented.")

def randomGaze():
    cam_name="camera"   #Creates an identifier for the camera subscription
    cam_type = 0        #0 for top camera, 1 for bottom camera
    res =1              # 320x240
    colspace=13         #BGR colorspace
    fps=10              #The requested frames per second

    cams = videoProxy.getSubscribers()
    for cam in cams:
        videoProxy.unsubscribe(cam)

    cam = videoProxy.subscribeCamera(cam_name,cam_type,res,colspace,fps)

    timeSinceStartMovement = time.time()
    while (time.time-timeSinceStartMovement)<20

        #image_container contains info about the image
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
        try:
            circle = circles[0,:][0]
            tts.say("I found the ball")
            return time.time()-timeSinceStartMovement
        except: #TODO: find the exception for not seeing blue ball.
            vertiRand = random.uniform(-0.5,0.5)
            horiRand = random.uniform(-0.5,0.5)
            motionProxy.angleInterpolation(headJointsHori, horiRand, [0.5], isAbsolute)
            motionProxy.angleInterpolation(headJointsVerti, vertiRand, [0.5], isAbsolute)
    tts.say("I could not find the ball")
    return 20

#End of randomGaze function.

if __name__ == "__main__":
    tts.say("Brooks framework demonstration start")
    tts.say("First, I would start training gaze detection.")
    tts.say("For now, I will skip that.")
    try:
        for i in range(5):
            findFace()
            choice = getChoice()
            if choice:
                result=faceGaze()
            else:
                result=randomGaze()
            tts.say("Time was")
            tts.say(result)
            tts.say("Trial done")
        postureProxy.goToPosture("Sit", 0.5)
        motionProxy.rest()
        pythonBroker.shutdown()
    except:
        postureProxy.goToPosture("Sit", 0.5)
        motionProxy.rest()
        pythonBroker.shutdown()
