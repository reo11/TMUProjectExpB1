function histFeature = funcHIST(img) 
    % DCT“Á’¥—Ê
    testImg = img(:); %
    histFeature = hist(double(testImg), 256);
end

