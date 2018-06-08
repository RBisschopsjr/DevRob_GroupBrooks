'''Testing of turnAngle.py before adding to the framework.py'''
import matlab.engine
import numpy as np
import cv2
import math

from turnAngle import getTurnAngle, adjust_gamma

print 'Starting Matalb...'
eng = matlab.engine.start_matlab()


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

face = [210,  91,  79,  79]
x = face[0]
y = face[1]
w = face[2]
h = face[3]

x_old = float(x)+(float(w)/2.0)
y_old = float(y)+(float(h)/2.0)

print 'x_old:',x_old, '     y_old:', y_old

x_old = float(x_old/320.0)
y_old = float(y_old/240.0)
print 'x_old:',x_old, '     y_old:', y_old

imageName = 'bt_faceimage.png'
# x_in_p = 0.378125
# y_in_p = 0.28333

x_in_p = x_old
y_in_p = y_old

print 'x_in_p:',x_in_p, '     y_in_p:', y_in_p

print 'Calling Gaze'
(y_out, x_out) = eng.callMatGaze(imageName, x_in_p, y_in_p, nargout=2)
print 'Gaze Pos:> x_new:',x_out, '     y_new:', y_out

print 'turn angle'
(thY, thX) = getTurnAngle(imageName, x_in_p, y_in_p, x_out, y_out, saveImg=False, showImg=False)
print thY, thX

(thY, thX) = getTurnAngle(imageName, 0.2, 0.2, 160, 120)
print thY, thX

'''
image brighness test
'''
# imgName = 'faceimage2.png'
# img = cv2.imread(imgName)
# imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# imageBT = adjust_gamma(imgRGB, gamma=1.5)
# cv2.imwrite('bt_'+imgName, cv2.cvtColor(imageBT, cv2.COLOR_RGB2BGR))
