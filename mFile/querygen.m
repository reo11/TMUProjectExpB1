Qpath = '../facedata/Query/jpeg/*.jpg';
path = '../facedata/Query/jpeg/';
D = dir(Qpath);
fileID = fopen('answer.csv','w');
str = {'query_name','answer'};
fprintf(fileID,'query_num,answer\n');
for i=1:length(D)
   name = strcat(path, D(i).name);
   img = imread(name);
   answer = extractBefore(extractAfter(name,'jpeg/'),3);
    if answer(1) == 'r';
        answer = 0;
    else
        answer = str2num(answer) + 1;
    end
    Query(:,:,i) = img;
    QueryAnswer(i) = answer;
    fprintf(fileID,'%s,%d\n',D(i).name ,answer);
end
fclose(fileID);