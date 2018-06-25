
useGazeServer = False
newFaceDect = True

if useGazeServer:
    from ServerGaze import getServerGaze
else:
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


if newFaceDect:
    import face_recognition

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

if not useGazeServer:
    print 'Starting Matlab ...'
    eng = matlab.engine.start_matlab()

print 'Starting Program ...'

class Agent:

    def __init__(self, policy_names):

	# Maximum time of looking around
        self.attention = 20
        self.policy_names = policy_names
        # Stores values to base decision on
        self.policy_values = [1.0 for _ in policy_names]


    '''
	Returns the probability of choosing each of its different policies as a list.
	This begins as a uniform distribution over the possible policies.
    '''
    def get_probs(self):
        return [x/sum(self.policy_values) for x in self.policy_values]


    '''
	Chooses a specific policy (as a string name) to enact pseudo-randomly,
	given its preferences.
    '''
    def get_policy(self):
        probs = self.get_probs()
        choice = np.random.uniform()
        for i, p in enumerate(probs):
            if p < choice:
                choice -= p
            else:
                return self.policy_names[i]
        return self.policy_names[-1]


    '''
	Updates the beliefs given an observations, which must be a probability vector as
	a list	with length equal to the number of policies.
    '''
    def update_policies(self, observations):
        if len(observations) != len(self.policy_values):
            print("Observation length must equal the mumber of different policies")
        else:
            self.policy_values = [x + y for x, y in zip(self.policy_values, observations)]

def time_to_observation(time, attention, nr_policies, index):

	time = max(0, min(attention, time))
	fitness = float(attention - time)/float(attention)

	observation = [(1-fitness)/(nr_policies-1) for x in range(nr_policies)]
	observation[index] = fitness

	return observation

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
    directionList = [[0.5, 0.5],[-0.5,0.5],[-0.5,-0.5],[0.5,-0.5]]
    index=0
    counter=0
    while True:
        # image_container contains info about the image
        image_container = videoProxy.getImageRemote(cam)

        # get image width and height
        width = image_container[0]
        height = image_container[1]

        # the 6th element contains the pixel data
        values = map(ord, list(image_container[6]))

        image = np.array(values, np.uint8).reshape((height, width, 3))

        imageName = 'faceimage_test.png'
        # print '++++++++ writing image'
        cv2.imwrite(imageName, image)

        ## New Face Detector
        if newFaceDect:
            # print 'Using New Face detector'
            image = face_recognition.load_image_file(imageName)
            faces = face_recognition.face_locations(image)
            print 'Faces detected:', faces
            if len(faces) > 0 :
                (face_y, face_right, face_bottom, face_x) = faces[0]
                face_w = face_right - face_x
                face_h = face_bottom - face_y

                tts.say("Found you")
                print "found You"
                return [face_x, face_y, face_w, face_h], None

        ## OpenCV Face detection
        else:
            print 'Using OpenCV Face detector'
            image = cv2.imread(imageName)
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
            if directionList[index][0]>0:
                horiRand = random.uniform(0,directionList[index][0])
            else:
                horiRand= random.uniform(directionList[index][0],0)
            if directionList[index][1]>0:
                vertiRand = random.uniform(0,directionList[index][1])
            else:
                vertiRand= random.uniform(directionList[index][1],0)
            motionProxy.angleInterpolation([headJointsHori,headJointsVerti], [horiRand, vertiRand], [0.5,0.5], isAbsolute)
            counter+=1
            if counter>3:
                counter=0
                index+=1
            if index>3:
                index=0

#TODO: Implement determining what behaviour to pick.
def getChoice():
    return True
    #return random.randint(0,1)==1


#TODO: Implement finding object through gaze.
def faceGaze(face):

    tts.say("Testing direction")
    print '\n=== get Gaze ===\n'
    begin_time = time.time()
    imgHeight = 240.0
    imgWidth = 320.0

    imgName = 'faceimage_test.png' # face detected Image
    isAbsolute=False
    '''
    Center on Face
    '''
    # tts.say("Centering Face")
    ang_x, ang_y = -(face[0]+face[2]/2-160.0)/320.0, (face[1]+face[3]/2-120.0)/240.0

    before_time = time.time()-begin_time


    print ang_x, ang_y
    print 'centering face'
    tts.say("centering face")


    begin_time = time.time()

    motionProxy.angleInterpolation([headJointsHori,headJointsVerti], [ang_x, ang_y], [0.5,0.5], isAbsolute)
    # save current image to be used for Gaze detector
    before_time += time.time()-begin_time
    ## face detect 2nd time
    tts.say("Looking for face again")
    newface, neweye = findFace(random_enable=False)
    print 'newFace:', newface

    begin_time = time.time()

    # newFace: [85 94 76 76]
    x = newface[0]
    y = newface[1]
    w = newface[2]
    h = newface[3]

    # img = cv2.imread(imgName)
    # plt.imshow(img)
    # x1, y1 = [x, x+w, x+w, x], [y, y, y+h, y+h]
    # plt.plot(x1, y1, marker = 'x')
    # # plt.show()
    # # plt.savefig('save_center_face.png')
    # plt.close()
    # print 'face detect saveds'

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

    imageBT = adjust_gamma(imgRGB, gamma=1.0)
    imgName = 'bt_'+imgName
    # save the new high brighness image as "bt_<imaName>"
    cv2.imwrite(imgName, cv2.cvtColor(imageBT, cv2.COLOR_RGB2BGR))

    if useGazeServer:
        print 'Using Gaze Server ...'
        (y_gaze, x_gaze) = getServerGaze(imgName, x_face_pct, y_face_pct)
        print 'Gaze Pos:> x_gaze:',x_gaze, '     y_gaze:', y_gaze
    else:
        print 'Local Gaze'
        (y_gaze, x_gaze) = eng.callMatGaze(imgName, x_face_pct, y_face_pct, nargout=2) # output - Y, X format
        print 'Gaze Pos:> x_gaze:',x_gaze, '     y_gaze:', y_gaze
    ### calculate the head turnAngle
    # x in, y in, x out, y out
    (turnAngle_y, turnAngle_x) = getTurnAngle(imgName, x_face_pct, y_face_pct, x_gaze, y_gaze, saveImg=True)



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
    turn_step_angle_x = float(turnAngle_x/5.0)
    turn_step_angle_y = float(turnAngle_y/5.0)
    print 'turn_step_angle_x:',turn_step_angle_x, '\t turn_step_angle_y:', turn_step_angle_y

    # return 20


    totalTurnedAngle_x = 0
    totalTurnedAngle_y = 0

    turn_count = 30

    x_limit_reached = False
    y_limit_reached = False

    time_taken = (time.time()-begin_time)+before_time

    tts.say("Following gaze")

    # time_taken = time.time() - begin_time
    while time_taken<20:
        print '*** time_taken:', time_taken

        print '\nturn_count::',turn_count
        tm1 = time.time()
        # commandAngles_y = motionProxy.getAngles(headJointsVerti, False)
        sensorAngles_y = motionProxy.getAngles(headJointsVerti, True)
        # commandAngles_x = motionProxy.getAngles(headJointsHori, False)
        sensorAngles_x = motionProxy.getAngles(headJointsHori, True)

        sensorAngles_x = sensorAngles_x[0]
        sensorAngles_y = sensorAngles_y[0]
        print 'X angle:',sensorAngles_x, '\t Y angle:',sensorAngles_y
        # turn one step in X & Y
        if not (sensorAngles_x > -1 and sensorAngles_x < 1):
            x_limit_reached = True
            turn_step_angle_x=0
        if not (sensorAngles_y > -1 and sensorAngles_y < 1):
            y_limit_reached = True
            turn_step_angle_y=0

        if y_limit_reached:
            turn_step_angle_y = 0.0

        if x_limit_reached:
            turn_step_angle_x = 0.0

        print 'turn_step_angle_x:',turn_step_angle_x, '\t turn_step_angle_y:',turn_step_angle_y
        motionProxy.angleInterpolation([headJointsHori,headJointsVerti], [turn_step_angle_x, turn_step_angle_y], [0.1, 0.1], isAbsolute)


        # Limit check
        if x_limit_reached and y_limit_reached:
            print "\n***INFO***: X and Y limits Reached\n"
            tts.say("turn limit")
            return 20

        ## TODO: Object detection for ball
        print 'Looing for ball...'
        time_before_ball = time.time() - tm1
        getBall = findBall()
        time_1 = time.time()
        if getBall is not None:
            if getBall[0] > 0.0 and getBall[1] > 0.0:
                tts.say("I found the ball")
                print '\nFound Ball'
                print getBall #(X, Y, radius)
                # TODO: Center on the ball
                # x, y, w, h
                # x, y, radius
                center_ball_x, center_ball_y = -(getBall[0]-160.0)/320.0, (getBall[1]-120.0)/240.0

##                center_ball_x, center_ball_y = -( ball_x + ball_w/2 - 160.0 )/320.0, (ball_y + ball_h/2 - 120.0)/240.0
                print '1. ball Pos. X ({}) Y ({})'.format(center_ball_x, center_ball_y)
                #(center_ball_y, center_ball_x) = getTurnAngle('ballimage_1.png', 0.5, 0.5, getBall[0], getBall[1])
                print '2. ball Pos. X ({}) Y ({})'.format(center_ball_x, center_ball_y)

                if center_ball_y>0.0:
                    print 'Turn Down:', center_ball_y
                else:
                    print 'Turn Up:', center_ball_y

                if center_ball_x>0.0:
                    print 'Turn Left:', center_ball_x
                else:
                    print 'Turn Right:', center_ball_x

                print 'centering Ball'
                tts.say("centering Ball")
                isAbsolute = False
                motionProxy.angleInterpolation([headJointsHori,headJointsVerti], [center_ball_x, center_ball_y], [0.5,0.5], isAbsolute)
                time_taken = (time.time()-time_1) + time_before_ball + time_taken
                print 'Time taken:', time_taken
                tts.say("Task Complete")
                # break
                return time_taken
            ###
        ###
        #safety
        if not turn_count:
            tts.say('count limit')
            break
        turn_count -= 1 ## DEBUG:
        # print 'turn_count::',turn_count

    print "\n Gaze follow END - Time limit reached..."
    return 20
'''
# DEBUG: images
1. 'faceimage_test.png'
2. 'faceCenterimage.png'
3. 'bt_faceCenterimage.png'
4. 'save_bt_faceCenterimage.png'
5. 'newPosition.png'
'''
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

def randomGaze():

    isAbsolute=True
    time_taken = 0
    while (time_taken)<20:
        print '*** time_taken:', time_taken
        #image_container contains iinfo about the image
        getBall = findBall()
        tm1 = time.time()
        if getBall is not None:
            if getBall[0] > 0.0 and getBall[1] > 0.0:
                tts.say("I found the ball")
                print '\nFound Ball'
                print getBall #(X, Y, radius)

                center_ball_x, center_ball_y = -(getBall[0]-160.0)/320.0, (getBall[1]-120.0)/240.0

                #(center_ball_y, center_ball_x) = getTurnAngle('ballimage_1.png', 0.5, 0.5, getBall[0], getBall[1])
                print '2. ball Pos. X ({}) Y ({})'.format(center_ball_x, center_ball_y)

                if center_ball_y>0.0:
                    print 'Turn Down:', center_ball_y
                else:
                    print 'Turn Up:', center_ball_y

                if center_ball_x>0.0:
                    print 'Turn Left:', center_ball_x
                else:
                    print 'Turn Right:', center_ball_x

                print 'centering Ball'
                tts.say("centering Ball")
                isAbsolute = False
                motionProxy.angleInterpolation([headJointsHori,headJointsVerti], [center_ball_x, center_ball_y], [0.5,0.5], isAbsolute)
                # time_taken = (time.time()-time_1) + time_before_ball


                tm2 = time.time()-tm1
                print 'Time taken:', time_taken+tm2
                tts.say("Task Complete")
                return time_taken+tm2
        else:
            vertiRand = random.uniform(-1,1)
            horiRand = random.uniform(-1,1)
            # motionProxy.angleInterpolation(headJointsHori, horiRand, [0.5], isAbsolute)
            # motionProxy.angleInterpolation(headJointsVerti, vertiRand, [0.5], isAbsolute)
            motionProxy.angleInterpolation([headJointsHori,headJointsVerti], [horiRand, vertiRand], [1.0, 1.0], isAbsolute)
        time_taken = time.time()-tm1+time_taken
    tts.say("I could not find the ball")
    print "I could not find the ball"
    return 20

def randomGaze_bk():
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
            # motionProxy.angleInterpolation(headJointsHori, horiRand, [0.5], isAbsolute)
            # motionProxy.angleInterpolation(headJointsVerti, vertiRand, [0.5], isAbsolute)
            motionProxy.angleInterpolation([headJointsHori,headJointsVerti], [horiRand, vertiRand], [0.5, 0.5], isAbsolute)
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
    tts.say("Starting Test")
    #tts.say("First, I would start training gaze detection.")
    #tts.say("For now, I will skip that.")
    # cv2.waitKey(0)

    postureProxy.goToPosture("Sit", 0.5)
    robot = Agent(["random", "gaze-directed"])

    robot.policy_values = [28.567540961527552, 36.43245903847245]
    print 'policy_values:',robot.policy_values
    robot.update_policies([0.439500630177347, 0.5604993698226531])
    beliefs =[robot.get_probs()]
    print 'start belief:',beliefs
    epochs = 2
    try:
        motionProxy.setStiffnesses(headJointsHori, 0.6) #Set stiffness of limbs.
        motionProxy.setStiffnesses(headJointsVerti,0.6)
        print "Loop Starting..."
        for i in range(epochs):
            print '\nIteration:',i
            tts.say("test "+str(i+1))
            face, eyes =findFace(random_enable=True)
            choice = robot.get_policy()
            print 'choice:', choice
            # tts.say(choice)
            # if choice=="gaze-directed":
            #     index=1
            #     result= faceGaze(face)
            # else:
            #     index=0
            #     result = randomGaze()
            result= faceGaze(face)
            # observation = time_to_observation(result,robot.attention,2,index)
            # robot.update_policies(observation)
            # beliefs.append(observation)
            # robot.update_policies(observation)
            # beliefs.append(robot.get_probs())

            postureProxy.goToPosture("Sit", 0.5)
            tts.say("Change ball position")
            time.sleep(3)

        tts.say("Trial done")
        print 'beliefs'
        print beliefs
        print 'policy_values:',robot.policy_values

        # plt.plot(beliefs)
        rdata = []
        gdata = []
        for [val1, val2] in beliefs:
            # print val1, val2
            rdata.append(val1)
            gdata.append(val2)
        # line_up, = plt.plot([1,2,3], label='Line 2')
        leg1, = plt.plot(rdata, label='Random')
        leg2, = plt.plot(gdata, label='Gaze')
        plt.legend(handles=[leg1, leg2])
        plt.xlabel("Epochs")
        plt.title("Nao Robot Testing results")
        # plt.xlim([0, epochs])
        plt.ylabel("P(gaze-directed)")
        # plt.ylim([0, 1])

        postureProxy.goToPosture("Sit", 0.5)
        motionProxy.rest()
        pythonBroker.shutdown()

        # plt.show()

    except Exception as e:
        print e
        postureProxy.goToPosture("Sit", 0.5)
        motionProxy.rest()
        pythonBroker.shutdown()
