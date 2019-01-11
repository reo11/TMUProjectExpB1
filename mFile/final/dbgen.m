% MATLAB上にDBを実装するためのスクリプトMファイル
clc
clear
c = 20; % クラス総数
n = 10; % 1クラス当たりの学習パターン数

% DBの画像ファイル場所
path = 'M:\project\dataset2\DB\jpeg\';

%H = fspecial('disk', 20); %フィルターの作成(ぼかし)
for i=1:c
    for j=1:n
        str = strcat(path, num2str(n*(i-1)+j-1, '%03d'), '.jpg');
        img = imread(str);
       
          %img = edge(img, 'Canny');
          %img = imfilter(img, H, 'replicate');

        DB(:,:,n*(i-1)+j) = img; % 3次元配列の場合
    end
end

%% クエリのDB
%クエリDBの画像場所
Qpath = 'M:\project\dataset2\Query\jpeg\*.jpg';
path = 'M:\project\dataset2\Query\jpeg\';

%Qpathの中身
D = dir(Qpath);

for i=1:length(D)
    name = strcat(path, D(i).name);
    img = imread(name);
    
    %img = edge(img, 'Canny');
    %img = imfilter(img, H, 'replicate');
    
    Query(:,:,i) = img;
end