% function makeSubspaceModel(DB, Query)
    faceClass = 20;
    number = 10;
    r=30;

    for i=1:faceClass
        for j=1:number
            normDB(:,:,j,i) = double(DB(:,:,number*(i-1)+j))./norm(double(DB(:,:,number*(i-1)+j)));
            [picX, picY] = size(DB(:,:,number*(i-1)+j));

            for k=1:picY
                for l=1:picX
                    arrangedDB(k*picY+picX,(i-1)+j) = normDB(picY,picX,j,i);
                end
            end
        end
    end
    [picX, picY] = size(DB(:,:,1));
    [picSize, picNum] = size(arrangedDB(:,:));
    W=zeros(picSize,100,faceClass);
    for i=1:faceClass
        dbsize = size(arrangedDB);
        for j=1:number
            if j == 1;
                X = arrangedDB(:,(i-1)+j);
            else
                X = horzcat(X,arrangedDB(:,(i-1)+j));
            end
        end
        C = X*X.';
        % eigenvalues ​​and eigenvectors of C
        [eig_vec, eig_val]=eig(C);
        [value index]=sort(-diag(eig_val));
        W(:,:,j+1)=eig_vec(:,index(1:100));
        fprintf('Face Class %d ... OK\n',i);
    end

    S=zeros(faceClass,1);
    CONF=zeros(faceClass,faceClass);
    querySize = size(Query);
    for j=1:querySize(3)
        normQuery(:,:,j,i) = double(Query(:,:,j))./norm(double(Query(:,:,j)));
        [picX, picY] = size(Query(:,:,j));
        for k=1:picY
            for l=1:picX
                arrangedQuery(k*picY+picX,j) = normDB(picY,picX,j);
            end
        end
    end

    for i = 1 : querySize(3)
        for j = 1 : faceClass, S(j+1)=sum((transpose(W(:,1:r,j))*arrangedQuery(:,i)).^2);, end
        [value index]=max(S);
        if QueryAnswer(i)>0
            CONF(QueryAnswer(i),index)=CONF(QueryAnswer(i),index)+1;
        end
        fprintf('test data %d\n',i);
    end
    accuracy=(sum(diag(CONF))./querySize(3)).*100;
    fprintf('accuracy=%3.2f%%\n',accuracy);
%end