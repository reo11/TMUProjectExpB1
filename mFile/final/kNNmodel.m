% DB: DBの特徴量
% X: DBの正解ラベル

function Mdl = kNNmodel(feature, label)
    
    Mdl = fitcknn(feature, label, 'NumNeighbors',5,'Standardize',1);
    
end