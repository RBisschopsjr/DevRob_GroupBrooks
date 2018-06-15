import matlab.engine

from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2


import numpy as np
import cv2


print 'Starting Matalb...'
eng = matlab.engine.start_matlab()
print 'Matlab Ready'

# Initialize the Flask application
app = Flask(__name__)


# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
    r = request
    x_in_p = float(r.args.get('x'))
    y_in_p = float(r.args.get('y'))
    print 'X:',x_in_p, '\t Y:',y_in_p

    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    imageName = 'new_server_img.jpg'
    cv2.imwrite(imageName, img)

    # imageName = 'test.jpg'

    # do some fancy processing here....

    # TODO: Gaze detection
    (y_out, x_out) = eng.callMatGaze(imageName, x_in_p, y_in_p, nargout=2)
    print 'Gaze Pos:> x_new:',x_out, '     y_new:', y_out

    # build a response dict to send back to client
    response = {'y': str(y_out),'x': str(x_out)}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
# app.run(host="0.0.0.0", port=5000)
app.run(host="192.168.1.119", port=6000)
