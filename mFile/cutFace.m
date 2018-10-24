% function arrangedDB = cutFace(DB){
    DBpath = '../dataset/DB/jpeg/*.jpg';
    path = '../dataset/DB/jpeg/';
    status = 1;
    foldername = strcat('../facedata/DB/jpeg/');
    [status, msg, msgID] = mkdir(foldername)

    faceDetector = vision.CascadeObjectDetector;
    dbname = dir(DBpath)
    for i=1:length(dbname)
        name = strcat(path, dbname(i).name)
        I = imread(name);
        bboxes = faceDetector(I);
        if isempty(bboxes);
            [IFaces, rect] = imcrop(I)
            rect(3) = rect(4);
            IFaces = imcrop(I,rect);
        elseif size(bboxes) > 1
            [IFaces, rect] = imcrop(I)
            rect(3) = rect(4);
            IFaces = imcrop(I,rect);
        else 
            bboxes(3) = bboxes(4);
            IFaces = imcrop(I,bboxes);
        end
        imwrite(IFaces, strcat(foldername, dbname(i).name));
    end
    imageName = dir(strcat(foldername,'*.jpg'));
    maxY = 0;
    for i=1:length(imageName)
        name = strcat(foldername, imageName(i).name)
        I = imread(name);
        maxI = size(I,1);
        if maxY < maxI
            maxY = maxI;
        end
    end
    for i=1:length(imageName)
        name = strcat(foldername, imageName(i).name)
        I = imread(name);
        IFaces = imresize(I, [64, NaN]);
        imwrite(IFaces, strcat(foldername, dbname(i).name));
    end


    Qpath = '../dataset/Query/jpeg/*.jpg';
    path = '../dataset/Query/jpeg/';
    status = 1;
    foldername = strcat('../facedata/Query/jpeg/');
    [status, msg, msgID] = mkdir(foldername)

    faceDetector = vision.CascadeObjectDetector;
    queryname = dir(Qpath)
    for i=1:length(queryname)
        name = strcat(path, queryname(i).name)
        I = imread(name);
        bboxes = faceDetector(I);
        if isempty(bboxes);
            [IFaces, rect] = imcrop(I);
            rect(3) = rect(4);
            IFaces = imcrop(I,rect);
        elseif size(bboxes) > 1
            [IFaces, rect] = imcrop(I);
            rect(3) = rect(4);
            IFaces = imcrop(I,rect);
        else 
            bboxes(3) = bboxes(4);
            IFaces = imcrop(I,bboxes);
        end
        imwrite(IFaces, strcat(foldername, queryname(i).name));
    end
    imageName = dir(strcat(foldername,'*.jpg'));
    for i=1:length(imageName)
        name = strcat(foldername, imageName(i).name)
        I = imread(name);
        IFaces = imresize(I, [64, NaN]);
        imwrite(IFaces, strcat(foldername, queryname(i).name));
    end
% }