import requests
import json
import cv2

# import time
# start_time = time.time()

imageName_1 = 'test.jpg'
x_in_p = 0.6
y_in_p = 0.2679

global serverURL
serverURL = 'http://192.168.0.103:5000'

def getServerGaze(imageName, x_val, y_val):
    queryURL = serverURL + '/api/test?x='+str(x_in_p)+'&y='+str(y_in_p)
    # print queryURL
    # prepare headers for http request
    content_type = 'image/jpeg'
    headers = {'content-type': content_type}

    img = cv2.imread(imageName)
    # encode image as jpeg
    _, img_encoded = cv2.imencode('.jpg', img)
    # send http request with image and receive response
    response = requests.post(queryURL, data=img_encoded.tostring(), headers=headers)
    # decode response

    # print 'Gaze data received'
    gaze_json = json.loads(response.text)
    x_new = float(gaze_json['x'])
    y_new = float(gaze_json['y'])
    # print x_new
    # print y_new
    return (y_new, x_new)
# print("--- %s seconds ---" % (time.time() - start_time))
# expected output: {u'message': u'image received. size=124x124'}

print getServerGaze(imageName_1, x_in_p, y_in_p)
