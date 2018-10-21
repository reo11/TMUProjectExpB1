Qpath = '../dataset/Query/jpeg/*.jpg';
path = '../dataset/Query/jpeg/';
D = dir(Qpath);
for i=1:length(D)
   name = strcat(path, D(i).name);
   img = imread(name);
   answer = extractBefore(extractAfter(name,'jpeg/'),3);
    if answer(1) == 'r';
        answer = -1;
    else 
        answer = str2num(answer) + 1;
    end
   Query(:,:,i) = img;
   QueryAnswer(i) = answer;
end