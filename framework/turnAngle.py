import numpy as np
import cv2
import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''
Func: getTurnAngle : Determine Turn Angle, Nao head
Param:
    image_path: path to the image file
    x_in_p: Point 1, x coordinate in percentage
    y_in_p: Point 1, y coordinate in percentage
    x_out: Point 2, x coordinate in pixels
    y_out: Point 2, y coordinate in pixels
    saveImg: Bool varible to indicate whether to save an image with 2 points (default : False)
             (save image name : 'save'+image_path)
Return: (turn_angle_Y, turn_angle_X)
'''
def getTurnAngle(image_path, x_in_p, y_in_p, x_out, y_out, saveImg = False):

    # Load an color image in grayscale
    img = cv2.imread(image_path)

    # dimensions
    height, width = img.shape[:2]
    # print 'dim W:',width, ' H:',height

    #Input
    x_in = np.multiply(width, x_in_p)
    y_in = np.multiply(height, y_in_p)
    # print 'input X:',x_in, ' Y:',y_in
    #center
    x_c = width/2
    y_c = height/2

    Ya1 = abs(y_c-y_in)
    Xa1 = abs(x_c-x_in)

    Ya2 = abs(y_c-y_out)
    Xa2 = abs(x_c-x_out)

    x_d = abs(x_out-x_in)
    y_d = abs(y_out-y_in)

    # print 'Ya1:\t',Ya1,'\t Xa1:\t',Xa1
    # print 'Ya2:\t',Ya2,'\t Xa2:\t',Xa2

    theta_1 = math.atan(float(Ya1)/float(Xa1))
    theta_2 = math.atan(float(Ya2)/float(Xa2))

    turn_angle_X = math.pi-theta_1-theta_2
    turn_angle_Y = theta_1+theta_2

    # print 'Old: (',y_in,',',x_in,')'
    # print 'New: (',y_out,',',x_out,')'
    # print '\n theta_1:\t',round(np.rad2deg(theta_1)),'\t theta_2:\t',round(np.rad2deg(theta_2))
    # print 'turn_angle_X:\t',round(np.rad2deg(turn_angle_X)),'\t turn_angle_Y:\t',round(np.rad2deg(turn_angle_Y))

    x_factor = float(x_d)/float(width)
    y_factor = float(y_d)/float(height)
    # print 'x_factor:\t',x_factor,'\t y_factor:\t',y_factor

    turn_angle_X = np.multiply(turn_angle_X, x_factor)
    turn_angle_Y = np.multiply(turn_angle_Y, y_factor)
    # print 'turn_angle_X:\t',round(np.rad2deg(turn_angle_X)),'\t turn_angle_Y:\t',round(np.rad2deg(turn_angle_Y))

    # print 'X turn direction:',(x_out-x_in)
    if (x_out-x_in)>0:
        print 'Turn Right :',turn_angle_X
    else:
        turn_angle_X = (turn_angle_X*-1)
        print 'Turn Left :',turn_angle_X

    # print 'Y turn direction:',(y_out-y_in)
    if (y_out-y_in)>0:
        print 'Turn Down :',turn_angle_Y
    else:
        turn_angle_Y = (turn_angle_Y*-1)
        print 'Turn Up :',turn_angle_Y


    if saveImg:
        #draw the graph
        plt.imshow(img)
        x1, y1 = [x_in, x_out], [y_in, y_out]
        plt.plot(x1, y1, marker = 'o')
        # plt.show()
        plt.savefig('save'+image_path)

    return (turn_angle_Y, turn_angle_X)
