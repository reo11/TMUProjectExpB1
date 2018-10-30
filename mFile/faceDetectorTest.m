I  = imread('../facedata/DB/jpeg/000.jpg');
% Detect faces
faceDetector = vision.CascadeObjectDetector;
bboxes = step(faceDetector, I);
% Select the first face
face = I(bboxes(1,2):bboxes(1,2)+bboxes(1,4),bboxes(1,1):bboxes(1,1)+bboxes(1,3));
% Detect SURF features
ftrs = detectSURFFeatures(face);
%Plot facial features.
imshow(face);hold on; plot(ftrs);