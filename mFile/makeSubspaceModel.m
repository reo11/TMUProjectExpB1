function makeSubspaceModel(DB, Q)
    faceClass = 20;
    number = 10;
    r=30;

    for i=1:faceClass
        for j=1:number
            normDB(:,:,j,i) = double(DB(:,:,number*(i-1)+j))./norm(double(DB(:,:,number*(i-1)+j)));
        end
    end
    [picX, picY] = size(DB(:,:,1));
    W=zeros(picX,100,faceClass);
    for i=1:faceClass
        for j=1:number
            % C = normDB(:,:,j,i)*transpose(normDB(:,:,j,i));
            C = normDB(:,:,j,i);
            % eigenvalues ​​and eigenvectors of C
            [eig_vec, eig_val]=eig(C);
            [value index]=sort(-diag(eig_val));
            W(:,:,i)=eig_vec(:,index(1:100));
        end
        fprintf('Face Class %d ... OK\n',i);
    end

    querySize = size(Q)
    for i = 1 : querySize(3)
        for j = 0 : faceClass, S(j+1)=sum((double(W(:,1:r,j+1))*double(Q(:,:,i))).^2);, end
        [value index]=max(S);
        CONF(test_label(i)+1,index)=CONF(test_label(i)+1,index)+1;
        fprintf('test data %d\n',i);
    end
end