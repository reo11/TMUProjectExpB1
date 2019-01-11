%%
for i=1:58
X=Query(:,:,i);
number(i)=dct_matching(DB,X);
end

%%
h1 = figure;
imshow('pout.tif');

X=Query(:,:,i);
imcontrast(h1)

%% for i=1:58
for i=1:58
X=Query(:,:,i);
number(i)=matching(DB,X);
end
Qlabel(number)

%%

%クエリに対する判定
correctNum = 0;
for i=1:length(D)
    testImg = Query(:,:,i); %クエリの取得
    
    histF = funcHIST(testImg);
    %dctF = funcDCT(testImg);
    
    answer(i) = predict(model, histF);
    
    if(answer(i) == Qlabels(i))
        correctNum = correctNum + 1;
    end
end

%%
A=csvread('knn_dct.dat');
B=csvread('knn_dct.dat');
C=csvread('knn_dct.dat');
D=A+B+C;
[E,I]=max(D);
I=I-1;
for i=1:58
if(E(i)<1.5)
    I(i)=21;
end
end