Qpath = 'dataset/Query/jpeg/*.jpg';
path = 'dataset/Query/jpeg/';
D = dir(Qpath);
for i=1:length(D)
   name = strcat(path, D(i).name);
   disp(name);
   img = imread(name);
   Query(:,:,i) = img;
end