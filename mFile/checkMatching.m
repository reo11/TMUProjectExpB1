% Sample code of check
dbgen;
querygen;
acc = 0;
count = 0;
fileID = fopen('../submission/simple_matching.csv','w');
fprintf(fileID,'query_name,answer\n');
for i=1:length(D)
    answer = matching(DB, Query(:,:,i));
    if check(QueryAnswer, i, answer); % Change here to test.
        count = count + 1;
    end
    fprintf(fileID,'%s,%d\n',D(i).name ,answer);
end
fclose(fileID);
acc = count / length(D) * 100;
sprintf('Accuracy is %f%%',acc)