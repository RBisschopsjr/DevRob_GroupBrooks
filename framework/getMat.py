import matlab.engine
eng = matlab.engine.start_matlab()
#X, Y

#eng.addpath(r'/home/sameera/Documents/DevRob_GIT/DevRob_GroupBrooks/Parts code/Gaze',nargout=0)
out = eng.callMatGaze('test.jpg', 0.6, 0.2679, nargout=2)
print out
