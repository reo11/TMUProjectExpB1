%% knnモデルの作成

[fDB, C] = labeling(DB); %特徴量のラベリング
model = fitcknn(fDB, C);
model.NumNeighbors = 5; 