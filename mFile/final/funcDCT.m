function dctFeature = funcDCT(img)
   % DCTÁ¥Ê
   img4 = dct2(double(img)); %2³DCT
   imgdctlow = img4(1:15, 1:15); %áüg¬ªÌ²«oµ
   newimg4 = reshape(imgdctlow, [1, 15*15]); %1³»
   dctFeature = log(abs(newimg4));
end
