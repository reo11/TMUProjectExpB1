# 概要
プロジェクト実験のB1班

# 開発環境
MATLAB R2018b

# ピクセル毎の距離の比較
```MATLAB
dbgen
querygen
X = Query(:, :, 1);
matching(DB, X)
```

# k-最近傍法
TODO
コードとしては
```MATLAB
dbgen
querygen
X = Query(:, :, 1);
knn(DB, X)
```
で実行したい。(knnはK Nearest Neighborの略)
`function knn(DB, X)`でファイルを分割しましょう。

# 部分空間法
TODO
k-最近傍法と同様に。関数名は`subapace`かな