function number = matching(DB,Query,q)
%単純マッチング関数(関数Mファイル)
%クエリ画像XとDBの画像をピクセル毎に比較し、二乗誤差が最も小さい人物を出力する
X = Query(:,:,q);
dblX = double(X); 
for i=1:200
    A = DB(:,:,i);
    dblA = double(A);
    D = (dblX-dblA).^2;
    distance(i) = sum(sum(D));
end

[minimum, index] = min(distance);

number = ceil(index/10)-1;
%sprintf('X is Person %d.',number)
end

