import matlab.engine
eng = matlab.engine.start_matlab()
#X, Y
out = eng.callMatGaze('test.jpg', 0.6, 0.2679, nargout=2)
print out
