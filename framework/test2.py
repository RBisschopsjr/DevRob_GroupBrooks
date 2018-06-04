'''Testing of turnAngle.py before adding to the framework.py'''
import matlab.engine
import numpy as np
import cv2
import math

from turnAngle import getTurnAngle, adjust_gamma

print 'Starting Matalb...'
# eng = matlab.engine.start_matlab()


# inppu: X, Y
# output: Y, X
# eng.addpath(r'/home/sameera/Documents/DevRob_GIT/DevRob_GroupBrooks/Parts code/Gaze',nargout=0)

'''
Test case 1:
image '5.jpg'
head pos X,Y = [0.54, 0.28]

Test case 2:
image 'test.jpg'
head pos X,Y = [0.6, 0.2679]

'''

# imageName = '5.jpg'
# x_in_p = 0.54
# y_in_p = 0.28

# imageName = 'test.jpg'
# x_in_p = 0.6
# y_in_p = 0.2679
#
# (y_out, x_out) = eng.callMatGaze(imageName, x_in_p, y_in_p, nargout=2)
# print y_out, x_out
#
# (thY, thX) = getTurnAngle(imageName, x_in_p, y_in_p, x_out, y_out, saveImg=True)
# # print thY, thX

'''
image brighness test
'''
imgName = 'faceimage2.png'
img = cv2.imread(imgName)
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

imageBT = adjust_gamma(imgRGB, gamma=1.5)
cv2.imwrite('bt_'+imgName, cv2.cvtColor(imageBT, cv2.COLOR_RGB2BGR))