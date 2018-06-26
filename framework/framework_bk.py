import cv2
import numpy as np
import matplotlib.pyplot as plt
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
global postureProxy

PORT=9559
IP="192.168.1.105"
memory = ALProxy("ALMemory", IP, PORT)
tts =naoqi.ALProxy("ALTextToSpeech", IP, PORT)
motionProxy = naoqi.ALProxy("ALMotion", IP, PORT)
headJointsHori = "HeadYaw"
headJointsVerti = "HeadPitch"
videoProxy = naoqi.ALProxy('ALVideoDevice', IP, PORT)
pythonBroker = ALBroker("pythonBroker", "0.0.0.0", 9600, IP, PORT)
postureProxy = naoqi.ALProxy("ALRobotPosture", IP, PORT)

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
	as list	with length equal to the number of policies.
    '''
    def update_policies(self, observations):
        if len(observations) != len(self.policy_values):
            print("Observation length must equal the mumber of different policies")
        else:		
            self.policy_values = [x + y for x, y in zip(self.policy_values, observations)] 


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

def findFace():
    cam = setUpCam()
    directionList = [[0.5, 0.5],[-0.5,0.5],[-0.5,-0.5],[0.5,-0.5]]
    index=0
    counter=3

    while True:

        for num in range(0,5):
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
    ##          for (x,y,w,h) in faces:
    ##              cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    ##              roi_gray = gray[y:y+h, x:x+w]
    ##              roi_color = image[y:y+h, x:x+w]
    ##              eyes = eye_cascade.detectMultiScale(roi_gray)
    ##              for (ex,ey,ew,eh) in eyes:
    ##                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                tts.say("Found you")

                try:
                    print faces[0],eyes[0]
                    return faces[0], eyes[0]
                except:
                    return faces[0], None
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


#TODO: Implement finding object through gaze.
def faceGaze(face):
    #tts.say("Testing direction")
    x,y=-(face[0]+face[2]/2-160.0)/320.0, (face[1]+face[3]/2-120.0)/240.0
    isAbsolute=False
    motionProxy.angleInterpolation(headJointsVerti, y, [0.5], isAbsolute)
    motionProxy.angleInterpolation(headJointsHori, x, [0.5], isAbsolute)
    return 20

def findBall():
    cam = setUpCam()
    for num in range(0,5):
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
            return circles[0,:][0]
    return false

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
            print (circle)
            tts.say("I found the ball")
            return time.time()-timeSinceStartMovement
        else: #TODO: find the exception for not seeing green ball.
            vertiRand = random.uniform(-0.5,0.5)
            horiRand = random.uniform(-0.5,0.5)
            motionProxy.angleInterpolation(headJointsHori, horiRand, [0.5], isAbsolute)
            motionProxy.angleInterpolation(headJointsVerti, vertiRand, [0.5], isAbsolute)
    tts.say("I could not find the ball")
    return 20

#End of randomGaze function.

def time_to_observation(time, attention, nr_policies, index):

    time = max(0, min(attention, time))
    fitness = float(attention - time)/float(attention)
    
    observation = [(1-fitness)/(nr_policies-1) for x in range(nr_policies)]
    observation[index] = fitness
    
    return observation


if __name__ == "__main__":
    #tts.say("Brooks framework demonstration start")
    #tts.say("First, I would start training gaze detection.")
    #tts.say("For now, I will skip that.")
    postureProxy.goToPosture("Sit", 0.5)
    robot = Agent(["random", "gaze-directed"])
    beliefs =[robot.get_probs()]
    epochs=10
    try:
        motionProxy.setStiffnesses(headJointsHori, 0.8) #Set stiffness of limbs.
        motionProxy.setStiffnesses(headJointsVerti,0.8)
        
        for _ in range(epochs):
            face, eyes =findFace()
            choice = robot.get_policy()
            if choice=="gaze-directed":
                index=1
                #result=faceGaze(face)
                result=10+random.randint(-5,5)
            else:
                index=0
                result=15+random.randint(-5,5)
                #result=randomGaze()
                
            observation = time_to_observation(result,robot.attention,2,index)
            #observation[index] = policy_eval
            robot.update_policies(observation)
            belief=robot.get_probs()
            beliefs.append(belief)
            
            #tts.say("Time was")
            #tts.say(str(round(result)))
            #tts.say("Trial done")
        print(beliefs)
##        plt.plot(beliefs)
##        plt.xlabel("Epochs")
##        plt.title("Simulation without Nao where gaze-directed takes 10+-5 and random 15+-5 seconds")
##        plt.xlim([0, epochs])
##        plt.ylabel("P(gaze-directed)")
##        plt.ylim([0, 1])
##        plt.show()
        
        postureProxy.goToPosture("Sit", 0.5)
        motionProxy.rest()
        pythonBroker.shutdown()
    except Exception as e:
        print (e)
        postureProxy.goToPosture("Sit", 0.5)
        motionProxy.rest()
        pythonBroker.shutdown()
