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

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def calcTurnAngle(image_path, x_in_p, y_in_p, x_out, y_out):

    # Load an color image in grayscale
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    #dimensions
    print 'dim W:',width, ' H:',height
    #Input
    x_in = np.multiply(width, x_in_p)
    y_in = np.multiply(height, y_in_p)
    # print 'input X:',x_in, ' Y:',y_in

    #output
    # y_out = 388
    # x_out = 545
    # x_out = 1800

    # print 'output X:',x_out, ' Y:',y_out

    #centers
    x_c = float(width/2)
    y_c = float(height/2)
    # print 'center X:',x_c, ' Y:',y_c

    #length calc 1
    x_1 = abs(x_c - x_in)
    y_1 = abs(y_c - y_in)
    # print 'x_1:',x_1, ' y_1:',y_1
    L_1 = math.sqrt((x_1**2)+(y_1**2))
    # print 'L1:',L_1

    #angle 1
    ang_1 = math.acos(x_1/L_1)
    # print 'angle 1:',ang_1

    #length calc 2
    x_2 = abs(x_c - x_out)
    y_2 = abs(y_c - y_out)
    # print 'x_2:',x_2, ' y_2:',y_2
    L_2 = math.sqrt((x_2**2)+(y_2**2))
    # print 'L2:',L_2

    #angle 2
    ang_2 = math.acos(x_2/L_2)
    # print 'angle 2:',ang_2

    #turn angle
    turn_angle = float((math.pi - ang_1 - ang_2)/2.0)

    #get turn getDirection
    output_angle = 0
    print 'test dir:',(x_out-x_in)
    if (x_out-x_in)>0:
        output_angle = turn_angle
        print '\nTurn Left'
        print 'turn angle:',output_angle
    else:
        output_angle = (turn_angle*-1)
        print '\nTurn Right'
        print 'turn angle:',output_angle

    #draw the graph
    plt.imshow(img)
    x1, y1 = [x_in, x_out], [y_in, y_out]
    plt.plot(x1, y1, marker = 'o')
    # plt.show()
    plt.savefig('foo.png')

    return output_angle

# calcTurnAngle('test.jpg', 0.6, 0.2679, 545, 388)

# # Load an color image in grayscale
# img = cv2.imread('test.jpg')
# height, width = img.shape[:2]
# # del img
#
# #dimensions
# print 'dim W:',width, ' H:',height
#
# #Input
# x_in = np.multiply(width, 0.6)
# y_in = np.multiply(height, 0.2679)
#
# print 'input X:',x_in, ' Y:',y_in
#
# #output
# y_out = 388
# x_out = 545
# # x_out = 1800
#
# print 'output X:',x_out, ' Y:',y_out
#
# #centers
# x_c = float(width/2)
# y_c = float(height/2)
# print 'center X:',x_c, ' Y:',y_c
#
# #length calc 1
# x_1 = abs(x_c - x_in)
# y_1 = abs(y_c - y_in)
# print 'x_1:',x_1, ' y_1:',y_1
# L_1 = math.sqrt((x_1**2)+(y_1**2))
# print 'L1:',L_1
#
# #angle 1
# ang_1 = math.acos(x_1/L_1)
# print 'angle 1:',ang_1
#
# #length calc 2
# x_2 = abs(x_c - x_out)
# y_2 = abs(y_c - y_out)
# print 'x_2:',x_2, ' y_2:',y_2
# L_2 = math.sqrt((x_2**2)+(y_2**2))
# print 'L2:',L_2
#
# #angle 2
# ang_2 = math.acos(x_2/L_2)
# print 'angle 2:',ang_2
#
# #turn angle
# turn_angle = math.pi - ang_1 - ang_2
#
#
# #get turn getDirection
# if x_out-x_in>0:
#     print '\nTurn Left'
#     print '\nturn angle:',(turn_angle*-1)
# else:
#     print '\nTurn Right'
#     print '\nturn angle:',turn_angle
#
#
# #draw the graph
# plt.imshow(img)
# x1, y1 = [x_in, x_out], [y_in, y_out]
# plt.plot(x1, y1, marker = 'o')
# # plt.show()
# plt.savefig('foo.png')
#
# print "Rest"
