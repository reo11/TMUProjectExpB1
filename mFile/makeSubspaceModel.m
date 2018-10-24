function makeSubspaceModel(DB)
    faceClass = 20;
    number = 10;
    
    for i=1:faceClass
        for j=1:number
            normDB(:,:,number*(i-1)+j) = DB(:,:,number*(i-1)+j)./norm(DB(:,:,number*(i-1)+j));
        end
    end
end