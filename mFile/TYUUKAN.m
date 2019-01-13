clc
clear
%% DBの実装
c = 20; % クラス総数
n = 10; % 1クラス当たりの学習パターン数

% DBの画像ファイル場所
path = 'M:\project\dataset2\DB\jpeg\';
path2= 'M:\project\dataset2\DB\canny\';

%H = fspecial('disk', 20); %フィルターの作成(ぼかし)
for i=1:c
    for j=1:n
        str = strcat(path, num2str(n*(i-1)+j-1, '%03d'), '.jpg');
        img = imread(str);
       % img = imresize(img,[64 NaN]);
       % img = edge(img, 'Canny',[],2);
       %img = imfilter(img, H, 'replicate');
       %saveFile = strcat(path2, num2str(n*(i-1)+j-1, '%03d'), '.jpg');
       %imwrite(img, saveFile);
        DB(:,:,n*(i-1)+j) = img; % 3次元配列の場合
    end
end

%クエリのDB
%クエリDBの画像場所
Qpath = 'M:\project\dataset2\Query\jpeg\*.jpg';
path =  'M:\project\dataset2\Query\jpeg\';
path2 = 'M:\project\dataset2\Query\canny\';
%Qpathの中身
D = dir(Qpath);

for i=1:length(D)
    name = strcat(path, D(i).name);
    img = imread(name);
    csvwrite('test.csv',img);
    
   % img = imresize(img,[64 NaN]);
    img = edge(img, 'Canny',[],2);

   %saveFile = strcat(path2, D(i).name);
   %imwrite(img, saveFile);
   %img = imfilter(img, H, 'replicate');
    
    Query(:,:,i) = img;
end

%% 使用する機能選択
fprintf('どのアルゴリズムを使用しますか?\n')
prompt = '1=ピクセルマッチング,2=DCTマッチング,3=KNN(hist),4=KNN(dct),5=KNN(face_parts),6=KNN(HOG),7=部分空間(face_parts),8=部分空間(dct),9=部分空間(HOG)\n';
hanbetu = input(prompt);
%% ピクセルマッチング(精度50%[cannyなし,フィルターなし])
if(hanbetu == 1)
for i=1:length(D)
    answer(i) = matching(DB,Query,i);
end
end
%% DCTマッチング (低周波成分1:15で精度52%,エッジ検出適用で精度83%[cannyあり,フィルターなし])
if(hanbetu == 2)
  for i=1:58
        img = Query(:,:,i);
        img4 = dct2(double(img));
        imgdctlow = img4(1:15,1:15);
        %newimg4 = reshape(imgdctlow, [1, 15*15]);
        %UPP(i,:)=newimg4;
        DCT_DB(:,:,i)=imgdctlow ;
end
    %csvwrite('canny_dct15_Query.csv',UPP)
    tenpest=0;
for i=1:length(D)
        [answer(i),rjflag] = dct_matching(DCT_DB,Query,i);
        tenpest=tenpest+rjflag;
end
end
%% KNN
if(hanbetu >= 3 && hanbetu <=6)

%knnモデルの作成
[fDB, C] = labeling(DB,hanbetu); %特徴量のラベリング
model = fitcknn(fDB, C);
model.NumNeighbors = 5; 

%knnモデルによる回答
if(hanbetu == 5)
        Query_filename ='M:\project\dataset4\Query\csv\QFP.csv';
        Query_feature = csvread(Query_filename);
end

for i=1:length(D)
    testImg = Query(:,:,i); %クエリの取得
    
    if(hanbetu == 3)
        dctF = funcDCT(testImg);
        answer(i) = predict(model, dctF);
    elseif(hanbetu == 4)
        histF = funcHIST(testImg);
        answer(i) = predict(model, histF);
    elseif(hanbetu == 5)
        faceF = Query_feature(i,:);
        answer(i) = predict(model, faceF);
    elseif(hanbetu == 6)
        hogF = funcHOG(testImg);
        answer(i) = predict(model, hogF);
    end

end
     end


%% 部分空間(顔のパーツを特徴量として扱う場合)
if(hanbetu==7)
 %DB部分
    DB_filename='M:\project\dataset4\DB\csv\FP.csv';
    db_feature_list = csvread(DB_filename); 
    feature_num=20;
    
 %クエリ部分%
    Query_filename ='M:\project\dataset4\Query\csv\QFP.csv';
    Query_feature = csvread(Query_filename);
   %クエリのサイズ計算
    for i=1:58
        Query_feature(i,feature_num+1)=sqrt(sumsqr(Query_feature(i,:)));
    end
end
%% 部分空間(DCTを特徴量とする場合)
if(hanbetu==8)
 %DB部分
    for i=1:200
        img = DB(:,:,i);
        db_feature_list(i,:)=funcDCT(img);
    end
    feature_num=25;
   
 %クエリ部分
    for i=1:58
    img = Query(:,:,i);
    Query_feature(i,:)=funcDCT(img);
    end
for i=1:58
    Query_feature(i,feature_num+1)=sqrt(sumsqr(Query_feature(i,:)));
end
end
%% 部分空間(HOGを特徴量とする場合)
if(hanbetu==9)
        for i=1:200
        img = DB(:,:,i);
        db_feature_list(i,:)=funcHOG(img);
    end
    feature_num=2916;
     %クエリ部分
    for i=1:58
    img = Query(:,:,i);
    Query_feature(i,:)=funcHOG(img);
    end
for i=1:58
    Query_feature(i,feature_num+1)=sqrt(sumsqr(Query_feature(i,:)));
end
end
%% 部分空間回答部
if(hanbetu >= 7 && hanbetu <=9)
      %平均値計算
    sub_space=zeros(20,feature_num+1);
    for i=1:20 %i=人数
        for j=1:10 % i人目について10枚ずつ処理
            for k=1:feature_num %特徴量が20個
                test(j,k)=db_feature_list((i-1)*10+j,k);
            end
        end
        %平均値をとる
        M=mean(test);
        for k=1:feature_num
        sub_space(i,k)=M(k);
        end
        zettai=sqrt(sumsqr(sub_space(i,:)));
        sub_space(i,feature_num+1)=zettai;
    end
    
hairetu=zeros(58,20);
    for j=1:58
    for i=1:20
        for k=1:feature_num
        hairetu(j,i)=hairetu(j,i)+Query_feature(j,k)*sub_space(i,k);
        end
        hairetu(j,i)=acos(hairetu(j,i)/(Query_feature(j,feature_num+1)*sub_space(i,feature_num+1)));
    end
    end
    [S,answer] = min(hairetu,[],2);
for i=1:58
    answer(i)=answer(i)-1;
end

end
%% Answer check

% Qラベルの作成
A = zeros(1, 8);
Qlabels = A;
nums = [3, 3, 4, 3, 3, 4, 2, 3, 3, 3, 3, 3, 3, 2, 1, 1, 1, 1, 2, 2];
for i=1:20
    A = ones(1, nums(i) ) * i;
    Qlabels = horzcat(Qlabels, A);
end

% 正解率の判定
correctNum = 0;
for i=1:length(D)
    if(answer(i) == Qlabels(i))
        correctNum = correctNum + 1;
    end
end

sprintf('正解率: %.3f', correctNum / ( length(D) ) )
%%
