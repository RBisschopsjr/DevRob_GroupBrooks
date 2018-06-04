import numpy as np
import cv2
import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = cv2.imread('faceimage2.png')
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

hsv1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv3 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

addVal = 10
rows, columns =  np.where(hsv1[:,:,2]<(255-addVal))
for x in range(len(rows)):
    hsv1[:,:,2][rows[x]][columns[x]] += addVal

addVal = 20
rows, columns =  np.where(hsv2[:,:,2]<(255-addVal))
for x in range(len(rows)):
    hsv2[:,:,2][rows[x]][columns[x]] += addVal

addVal = 30
rows, columns =  np.where(hsv3[:,:,2]<(255-addVal))
for x in range(len(rows)):
    hsv3[:,:,2][rows[x]][columns[x]] += addVal

img3 = cv2.cvtColor(hsv1, cv2.COLOR_HSV2RGB)
img4 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2RGB)
img5 = cv2.cvtColor(hsv3, cv2.COLOR_HSV2RGB)


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


# loop over various values of gamma
# for gamma in np.arange(1.0, 2.5, 0.5):
# 	# ignore when gamma is 1 (there will be no change to the image)
# 	if gamma == 1:
# 		continue
#
# 	# apply gamma correction and show the images
# 	gamma = gamma if gamma > 0 else 0.1
# 	adjusted = adjust_gamma(original, gamma=gamma)
# 	cv2.putText(adjusted,
#                 "g={}".format(gamma),
#                 (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 (0, 0, 255),
#                 3 )
# 	cv2.imshow("Images", np.hstack([original, adjusted]))
# 	cv2.waitKey(0)

image1 = adjust_gamma(imgRGB, gamma=1.5)
image2 = adjust_gamma(imgRGB, gamma=2.0)


plt.subplot(221)
plt.imshow(imgRGB)


plt.subplot(222)
plt.imshow(image1)

plt.subplot(223)
plt.imshow(image2)

plt.subplot(224)
plt.imshow(img5)

plt.show()
