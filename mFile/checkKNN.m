%% クエリlabelの作成
A = zeros(1, 8);
Qlabels = A;
nums = [3, 3, 4, 3, 3, 4, 2, 3, 3, 3, 3, 3, 3, 2, 1, 1, 1, 1, 2, 2];
for i=1:20
    A = ones(1, nums(i))*i;
    Qlabels = horzcat(Qlabels, A);
end
%% 正解率の判定


for i=1:length(D)
    testImg = Query(:,:,i); %クエリの取得
    
    histF = funcHIST(testImg);
    %dctF = funcDCT(testImg);
    
    answer(i) = predict(model, histF);
end


correctNum = 0;
for i=1:length(D)
    if(answer(i) == Qlabels(i))
        correctNum = correctNum + 1;
    end
end

sprintf('正解率: %.2f', correctNum/length(D))