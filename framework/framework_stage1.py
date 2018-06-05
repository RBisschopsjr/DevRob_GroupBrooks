import matlab.engine

import cv2
import numpy as np
import naoqi
import time
import random
from naoqi import ALModule
from naoqi import ALProxy
from naoqi import ALBroker
from turnAngle import getTurnAngle, adjust_gamma

# global eng
print 'Starting Matlab ...'
eng = matlab.engine.start_matlab()

global IP
global PORT
global memory
global tts
global motionProxy
global headJointsHori
global headJointsVerti
global videoProxy
global pythonBroker
global postureProxy



PORT=9559
IP="192.168.1.102"
memory = ALProxy("ALMemory", IP, PORT)
tts = naoqi.ALProxy("ALTextToSpeech", IP, PORT)
motionProxy = naoqi.ALProxy("ALMotion", IP, PORT)
headJointsHori = "HeadYaw"
headJointsVerti = "HeadPitch"
videoProxy = naoqi.ALProxy('ALVideoDevice', IP, PORT)
pythonBroker = ALBroker("pythonBroker", "0.0.0.0", 9600, IP, PORT)
postureProxy = naoqi.ALProxy("ALRobotPosture", IP, PORT)

##for c number of trials:
##	while not face found:
##		findFace()
##	random= takeRandomNumberOnPreference
##	PerformBehaviour(random)
##	state if ball was found
##shut down

def setUpCam():
    cam_name = "camera"  # Creates an identifier for the camera subscription
    cam_type = 0  # 0 for top camera, 1 for bottom camera
    res = 1  # 320x240
    colspace = 13  # BGR colorspace
    fps = 10  # The requested frames per second

    cams = videoProxy.getSubscribers()
    for cam in cams:
        videoProxy.unsubscribe(cam)

    cam = videoProxy.subscribeCamera(cam_name, cam_type, res, colspace, fps)
    return cam

def takePicture():
    cam = setUpCam()
    # image_container contains info about the image
    image_container = videoProxy.getImageRemote(cam)
    # get image width and height
    width = image_container[0]
    height = image_container[1]
    # the 6th element contains the pixel data
    values = map(ord, list(image_container[6]))
    image = np.array(values, np.uint8).reshape((height, width, 3))
    cv2.imwrite("newPosition.png", image)

def findFace():
    cam = setUpCam()

    while True:
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

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

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
            # tts.say("Found you")
            print "found You"
            try:
                return faces[0], eyes[0]
            except:
                return faces[0], None
        isAbsolute=True
        vertiRand = random.uniform(-0.5,0.5)
        horiRand = random.uniform(-0.5,0.5)
        motionProxy.angleInterpolation(headJointsHori, horiRand, [0.5], isAbsolute)
        motionProxy.angleInterpolation(headJointsVerti, vertiRand, [0.5], isAbsolute)

#TODO: Implement determining what behaviour to pick.
def getChoice():
    return True
    #return random.randint(0,1)==1

#TODO: Implement finding object through gaze.
def faceGaze(face):
    # tts.say("Testing direction")
    print 'get Gaze'
    # face : x, y, w, h
    # 121, 68, 82, 82
    ### calc face position in percentage
    x = face[0]
    y = face[1]
    w = face[2]
    h = face[3]

    x_old = float(x)+(float(w)/2.0)
    y_old = float(y)+(float(h)/2.0)

    print 'Head Pos   :> x:',x_old, '     y:', y_old

    x_old = float(x_old/320.0)
    y_old = float(y_old/240.0)
    print 'Head Pos(%):> x:',x_old, '     y:', y_old

    ### change image brighness before inserting to Gaze detector
    imgName = 'faceimage.png'
    img = cv2.imread(imgName)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    imageBT = adjust_gamma(imgRGB, gamma=1.5)
    imgName = 'bt_'+imgName
    # save the new high brighness image as "bt_<imaName>"
    cv2.imwrite(imgName, cv2.cvtColor(imageBT, cv2.COLOR_RGB2BGR))

    ### Call Gaze detector with high brightness Image
    #output - Y, X format
    # print imgName
    (y_new, x_new) = eng.callMatGaze(imgName, x_old, y_old, nargout=2)
    print 'Gaze Pos:> x_new:',x_new, '     y_new:', y_new

    ### calculate the head turnAngle
    # x in, y in, x out, y out
    (turnAngleY, turnAngleX) = getTurnAngle(imgName, x_old, y_old, x_new, y_new, saveImg=True)
    isAbsolute=False
    motionProxy.angleInterpolation(headJointsVerti, turnAngleY, [1.0], isAbsolute)
    motionProxy.angleInterpolation(headJointsHori, turnAngleX, [1.0], isAbsolute)
    return 20

def randomGaze():
    cam = setUpCam()

    timeSinceStartMovement = time.time()
    isAbsolute=True
    while (time.time()-timeSinceStartMovement)<20:
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

        lower_green = np.array([36,100,100], dtype = np.uint8)
        upper_green = np.array([86,255,255], dtype=np.uint8)

        #convert to a hsv colorspace
        hsvImage=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        #Create a treshold mask
        color_mask=cv2.inRange(hsvImage,lower_green,upper_green)

        #apply the mask on the image
        green_image = cv2.bitwise_and(image,image,mask=color_mask)

        kernel=np.ones((9,9),np.uint8)
        #Remove small objects
        opening =cv2.morphologyEx(color_mask,cv2.MORPH_OPEN,kernel)
        #Close small openings
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        #Apply a blur to smooth the edges
        smoothed_mask = cv2.GaussianBlur(closing, (9,9),0)

        #Apply our (smoothend and denoised) mask
        #to our original image to get everything that is blue.
        green_image = cv2.bitwise_and(image,image,mask=smoothed_mask)

        #Get the grayscale image (last channel of the HSV image
        gray_image = green_image[:,:,2]

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
        if circles is not None:
            circle = circles[0,:][0]
            # tts.say("I found the ball")
            print "I found the ball"
            return time.time()-timeSinceStartMovement
        else: #TODO: find the exception for not seeing green ball.
            vertiRand = random.uniform(-0.5,0.5)
            horiRand = random.uniform(-0.5,0.5)
            motionProxy.angleInterpolation(headJointsHori, horiRand, [0.5], isAbsolute)
            motionProxy.angleInterpolation(headJointsVerti, vertiRand, [0.5], isAbsolute)
    # tts.say("I could not find the ball")
    print "I could not find the ball"
    return 20

#End of randomGaze function.

def getDirection(x,y):
    while x>1 or y>1 or x<-1 or y<-1:
        x=x/10
        y=y/10
    return x, y

if __name__ == "__main__":
    #tts.say("Brooks framework demonstration start")
    #tts.say("First, I would start training gaze detection.")
    #tts.say("For now, I will skip that.")
    # cv2.waitKey(0)
    postureProxy.goToPosture("Sit", 0.5)
    try:
        motionProxy.setStiffnesses(headJointsHori, 0.8) #Set stiffness of limbs.
        motionProxy.setStiffnesses(headJointsVerti,0.8)

        for i in range(1):
            face, eyes =findFace()
            print face
            print eyes
            choice = getChoice()
            if choice:
                result=faceGaze(face)
            else:
                result=randomGaze()
            # tts.say("Time was")
            # tts.say(str(round(result)))
            # tts.say("Trial done")
            print "time was"
            print round(result)
            print "Trial done"
            takePicture()

        # time.sleep(5)
        # print 'Turning to new position'
        # isAbsolute=False
        # motionProxy.angleInterpolation(headJointsVerti, -1.0, [1.0], isAbsolute)
        # motionProxy.angleInterpolation(headJointsHori, -1.0, [1.0], isAbsolute)

        time.sleep(5)
        print 'done'
        postureProxy.goToPosture("Sit", 0.5)
        motionProxy.rest()
        pythonBroker.shutdown()
    except Exception as e:
        print e
        postureProxy.goToPosture("Sit", 0.5)
        motionProxy.rest()
        pythonBroker.shutdown()
