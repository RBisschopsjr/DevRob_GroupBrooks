 % Demo script to use the gaze following model
 % Written by Adria Recasens (recasens@mit.edu)
 
 %addpath(genpath('/data/vision/torralba/datasetbias/caffe-cudnn3/matlab/'));
 addpath(genpath('/home/sameera/Downloads/caffe-master/matlab'));
 
 im = imread('test.jpg');
 
 [H,W,D] = size(im);
 %Plug here your favorite head detector
 e = [0.6  0.2679];
 
 % format - Y, X
 floor([H*e(1,2),W*e(1,1)])
 % Compute Gaze 
 [x_predict,y_predict,heatmap,net] = predict_gaze(im,e);
 
 % format - Y, X
 outY=0;
 outX=0;
 output = floor([x_predict*H, W*y_predict ]);
 outY = output(1,1)
 outX = output(1,2)
 %Visualization
 g = floor([x_predict y_predict].*[size(im,2) size(im,1)]);
 e = floor(e.*[size(im,2) size(im,1)]);
 line = [e(1) e(2) g(1) g(2)];
 im = insertShape(im,'line',line,'Color','red','LineWidth',8);    
 
 image(im);
 
 
 
 
