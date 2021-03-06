function [fDB,C]=labeling(DB,hanbetu)
    %特徴量のDB作成関数

   if(hanbetu==5)
             DB_filename='M:\project\dataset4\DB\csv\FP.csv';
             feature_list = csvread(DB_filename); 
   end

    for i=1:200
         %DBのi枚目を読み込む
         img = DB(:,:,i);
         
         %i枚目の特徴量計算
         if(hanbetu == 3)
             feature = funcDCT(img); %dctの場合
         elseif(hanbetu == 4)
             feature = funcHIST(img); %histの場合
         elseif(hanbetu == 5)
             feature = feature_list(i,:);
         elseif(hanbetu == 6)
             feature = funcHOG(img);
         end
         
         %特徴量のDB
         fDB(:,i) = feature;
         
         %正解ラベル
         C(i) = fix((i-1)/10);
    end
    
    %行と列の入れ替え
    C = transpose(C);
    fDB = transpose(fDB);
end

