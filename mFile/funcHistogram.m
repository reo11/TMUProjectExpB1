% ヒストグラム特徴量の検出
function histFeature = funcHistogram(img)
    % 1次元化
    img = img(:);
    histFeature = hist(double(img),256);
end


