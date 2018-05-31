# import matlab.engine
# eng = matlab.engine.start_matlab()


#inppu: X, Y
#output: Y, X
#eng.addpath(r'/home/sameera/Documents/DevRob_GIT/DevRob_GroupBrooks/Parts code/Gaze',nargout=0)
# out = eng.callMatGaze('test.jpg', 0.6, 0.2679, nargout=2)
# print out

import numpy as np
import cv2
import math
# Load an color image in grayscale
img = cv2.imread('test.jpg')
height, width = img.shape[:2]
del img

#dimensions
print 'dim W:',width, ' H:',height

#Input
x_in = np.multiply(width, 0.6)
y_in = np.multiply(height, 0.2679)

print 'input X:',x_in, ' Y:',y_in

#output
y_out = 388
# x_out = 545
x_out = 1800

print 'output X:',x_out, ' Y:',y_out

#centers
x_c = float(width/2)
y_c = float(height/2)
print 'center X:',x_c, ' Y:',y_c

#length calc 1
x_1 = abs(x_c - x_in)
y_1 = abs(y_c - y_in)
print 'x_1:',x_1, ' y_1:',y_1
L_1 = math.sqrt((x_1**2)+(y_1**2))
print 'L1:',L_1

#angle 1
ang_1 = math.acos(x_1/L_1)
print 'angle 1:',ang_1

#length calc 2
x_2 = abs(x_c - x_out)
y_2 = abs(y_c - y_out)
print 'x_2:',x_2, ' y_2:',y_2
L_2 = math.sqrt((x_2**2)+(y_2**2))
print 'L2:',L_2

#angle 2
ang_2 = math.acos(x_2/L_2)
print 'angle 2:',ang_2

#turn angle
turn_angle = math.pi - ang_1 - ang_2


#get turn getDirection
if x_out-x_in>0:
    print '\nturn angle:',(turn_angle*-1)
else:
    print '\nturn angle:',turn_angle
