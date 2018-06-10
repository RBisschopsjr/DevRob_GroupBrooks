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

import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# global eng
## sudo iwconfig wlp4s0 power off

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

print cv2.__version__

PORT=9559
IP="192.168.1.103"
pythonBroker = ALBroker("pythonBroker", "0.0.0.0", 9600, IP, PORT)
memory = ALProxy("ALMemory", IP, PORT)
tts = naoqi.ALProxy("ALTextToSpeech", IP, PORT)
motionProxy = naoqi.ALProxy("ALMotion", IP, PORT)
headJointsHori = "HeadYaw"
headJointsVerti = "HeadPitch"
videoProxy = naoqi.ALProxy('ALVideoDevice', IP, PORT)

postureProxy = naoqi.ALProxy("ALRobotPosture", IP, PORT)

print 'Starting Matlab ...'
eng = matlab.engine.start_matlab()

print 'Starting Program ...'
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

def takePicture(filename):
    cam = setUpCam()
    # image_container contains info about the image
    image_container = videoProxy.getImageRemote(cam)
    # get image width and height
    width = image_container[0]
    height = image_container[1]
    # the 6th element contains the pixel data
    values = map(ord, list(image_container[6]))
    image = np.array(values, np.uint8).reshape((height, width, 3))
    cv2.imwrite(filename, image)

def findFace(random_enable):
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

        cv2.imwrite("faceimage_test.png", image)
        image = cv2.imread("faceimage_test.png")

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
            tts.say("Found you")
            print "found You"
            try:
                return faces[0], eyes[0]
            except:
                return faces[0], None
        if random_enable:
            isAbsolute=True
            vertiRand = random.uniform(-0.5,0.5)
            horiRand = random.uniform(-0.5,0.5)
            motionProxy.angleInterpolation(headJointsHori, horiRand, [0.5], isAbsolute)
            motionProxy.angleInterpolation(headJointsVerti, vertiRand, [0.5], isAbsolute)

#TODO: Implement determining what behaviour to pick.
def getChoice():
    return True
    #return random.randint(0,1)==1

def findBall():
    cam = setUpCam()

    #image_container contains iinfo about the image
    image_container = videoProxy.getImageRemote(cam)

    #get image width and height
    width=image_container[0]
    height=image_container[1]

    #the 6th element contains the pixel data
    values = map(ord,list(image_container[6]))

    image=np.array(values, np.uint8).reshape((height, width,3))

    cv2.imwrite("ballimage_1.png", image)
    image=cv2.imread("ballimage_1.png")

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
        return circles[0,:][0]
    else:
        return None

#TODO: Implement finding object through gaze.
def faceGaze(face):
    tts.say("Testing direction")
    print '\n=== get Gaze ===\n'
    imgHeight = 240.0
    imgWidth = 320.0

    imgName = 'faceimage_test.png' # face detected Image
    isAbsolute=False
    '''
    Center on Face
    '''
    # tts.say("Centering Face")
    ang_x, ang_y=-(face[0]+face[2]/2-160.0)/320.0, (face[1]+face[3]/2-120.0)/240.0
    print ang_x, ang_y
    print 'centering face'
    motionProxy.angleInterpolation(headJointsVerti, ang_y, [0.5], isAbsolute)
    motionProxy.angleInterpolation(headJointsHori, ang_x, [0.5], isAbsolute)
    # save current image to be used for Gaze detector

    ## face detect 2nd time
    tts.say("Looking for face again")
    newface, neweye = findFace(random_enable=False)
    print 'newFace:', newface
    # newFace: [85 94 76 76]
    x = newface[0]
    y = newface[1]
    w = newface[2]
    h = newface[3]

    img = cv2.imread(imgName)
    plt.imshow(img)
    x1, y1 = [x, x+w, x+w, x], [y, y, y+h, y+h]
    plt.plot(x1, y1, marker = 'x')
    # plt.show()
    plt.savefig('save_center_face.png')
    plt.close()
    print 'face detect saveds'

    x_face = float(x)+(float(w)/2.0)
    y_face = float(y)+(float(h)/2.0)
    print 'Head Pos   :> x:',x_face, '\t y:', y_face

    # position in percentage - needed for Gaze network
    x_face_pct = float(x_face/imgWidth)
    y_face_pct = float(y_face/imgHeight)
    print 'Head Pos(%):> x:',x_face_pct, '\t y:', y_face_pct

    imgName = 'faceCenterimage.png'
    takePicture(imgName)
    print "Face picture saved"
    # time.sleep(2)
    '''
    Gaze detection
    '''
    ### change image brighness before inserting to Gaze detector
    img = cv2.imread(imgName)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    imageBT = adjust_gamma(imgRGB, gamma=1.5)
    imgName = 'bt_'+imgName
    # save the new high brighness image as "bt_<imaName>"
    cv2.imwrite(imgName, cv2.cvtColor(imageBT, cv2.COLOR_RGB2BGR))

    ### Call Gaze detector with high brightness Image
    # TODO:
    # 1. Since now face is centered: x,y = 0.5 can be used
    # x_face_pct = 0.5
    # y_face_pct = 0.5
    # 2. Run face detection again and get the (x,y) of the face again

    (y_gaze, x_gaze) = eng.callMatGaze(imgName, x_face_pct, y_face_pct, nargout=2) # output - Y, X format
    print 'Gaze Pos:> x_gaze:',x_gaze, '     y_gaze:', y_gaze

    ### calculate the head turnAngle
    # x in, y in, x out, y out
    (turnAngle_y, turnAngle_x) = getTurnAngle(imgName, x_face_pct, y_face_pct, x_gaze, y_gaze, saveImg=True)

    '''
    # DEBUG: images
    1. 'faceimage_test.png'
    2. 'faceCenterimage.png'
    3. 'bt_faceCenterimage.png'
    4. 'save_bt_faceCenterimage.png'
    5. 'newPosition.png'
    '''

    '''
    Look at gaze position directly
    '''

    # motionProxy.angleInterpolation(headJointsVerti, turnAngle_y, [1.0], isAbsolute)
    # motionProxy.angleInterpolation(headJointsHori, turnAngle_x, [1.0], isAbsolute)

    '''
    Follow gaze step by step
    '''

    ### TODO: follow the gaze step by step
    # calc. angle per step
    turn_step_angle_x = float(turnAngle_x/10.0)
    turn_step_angle_y = float(turnAngle_y/10.0)
    print 'turn_step_angle_x:',turn_step_angle_x, '\t turn_step_angle_y:', turn_step_angle_y

    # return 20


    totalTurnedAngle_x = 0
    totalTurnedAngle_y = 0

    turn_count = 12
    # current_pos_x = 0.5
    # current_pos_y = 0.5
    #
    # turnLimit_x = 1
    # turnLimit_y = 1
    x_limit_reached = False
    y_limit_reached = False
    while True:
        # commandAngles_y = motionProxy.getAngles(headJointsVerti, False)
        sensorAngles_y = motionProxy.getAngles(headJointsVerti, True)
        # commandAngles_x = motionProxy.getAngles(headJointsHori, False)
        sensorAngles_x = motionProxy.getAngles(headJointsHori, True)

        sensorAngles_x = sensorAngles_x[0]
        sensorAngles_y = sensorAngles_y[0]
        print 'X angle:',sensorAngles_x, '\t Y angle:',sensorAngles_y
        # turn one step in X & Y
        if sensorAngles_x > -1 and sensorAngles_x < 1:
            print 'X step turn'
            motionProxy.angleInterpolation(headJointsHori, turn_step_angle_x, [0.2], isAbsolute)
        else:
            x_limit_reached = True

        if sensorAngles_y > -1 and sensorAngles_y < 1:
            print 'Y step turn'
            motionProxy.angleInterpolation(headJointsVerti, turn_step_angle_y, [0.2], isAbsolute)
        else:
            y_limit_reached = True

        # Limit check
        if x_limit_reached and y_limit_reached:
            print "\n***INFO***: X and Y limits Reached\n"
            break

        ## TODO: Object detection for ball
        getBall = findBall()
        if getBall is not None:
            if getBall[0] > 0.0 and getBall[1] > 0.0:
                print '\nFound Ball'
                print getBall #(X, Y, radius)
                # TODO: Center on the ball
                # ang_x, ang_y=-(face[0]+face[2]/2-160.0)/320.0, (face[1]+face[3]/2-120.0)/240.0
                # print ang_x, ang_y
                # print 'centering face'
                # motionProxy.angleInterpolation(headJointsVerti, ang_y, [0.5], isAbsolute)
                # motionProxy.angleInterpolation(headJointsHori, ang_x, [0.5], isAbsolute)
                (center_ball_y, center_ball_x) = getTurnAngle('ballimage_1.png', 0.5, 0.5, getBall[0], getBall[1])
                print 'centering Ball'
                tts.say("centering Ball")
                isAbsolute = False
                motionProxy.angleInterpolation(headJointsVerti, center_ball_y, [1.0], isAbsolute)
                motionProxy.angleInterpolation(headJointsHori, center_ball_x, [1.0], isAbsolute)
                tts.say("Task Complete")
                break

        #safety
        if not turn_count:
            break
        turn_count -= 1 ## DEBUG:
        print 'turn_count::',turn_count

        # totalTurnedAngle_x += turn_step_angle_x
        # totalTurnedAngle_y += turn_step_angle_y

        #Limit the maximum head turn - for safety of Nao
        ## TODO: get the current position angle from the absolute position
        # try using Nao API
        # getAngles
        # getLimits
        # getPosition

        # TODO: Calc current position (x,y) using above values

        # if current_pos_x >= turnLimit_x and current_pos_y >= turnLimit_y :
        #     # Speak: turn limit reached
        #     break # break when only both limits were reached


    #
    print "\n Gaze follow END"
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
        # cv2.HOUGH_GRADIENT
        circles = cv2.HoughCircles(#cv.CV_HOUGH_GRADIENT
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
            tts.say("I found the ball")
            print "I found the ball"
            return time.time()-timeSinceStartMovement
        else: #TODO: find the exception for not seeing green ball.
            vertiRand = random.uniform(-0.5,0.5)
            horiRand = random.uniform(-0.5,0.5)
            motionProxy.angleInterpolation(headJointsHori, horiRand, [0.5], isAbsolute)
            motionProxy.angleInterpolation(headJointsVerti, vertiRand, [0.5], isAbsolute)
    tts.say("I could not find the ball")
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
            face, eyes = findFace(random_enable=True)
            print 'face:', face
            print 'eyes:', eyes
            choice = getChoice()
            if choice:
                result=faceGaze(face)
            else:
                result=randomGaze()
            takePicture("newPosition.png")
            tts.say("Time was")
            tts.say(str(round(result)))
            tts.say("Trial done")
            print "time was"
            print round(result)
            print "Trial done"

        # time.sleep(5)
        # print 'Turning to new position'
        # isAbsolute=False
        # motionProxy.angleInterpolation(headJointsVerti, -1.0, [1.0], isAbsolute)
        # motionProxy.angleInterpolation(headJointsHori, -1.0, [1.0], isAbsolute)


        time.sleep(3)
        print 'done'
        postureProxy.goToPosture("Sit", 0.5)
        motionProxy.rest()
        pythonBroker.shutdown()
        sys.exit(0)
    except Exception as e:
        print e
        postureProxy.goToPosture("Sit", 0.5)
        motionProxy.rest()
        pythonBroker.shutdown()
        sys.exit(0)
