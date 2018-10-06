% Sample code of check
dbgen;
querygen;
acc = 0;
count = 0;
for i=1:length(D)
    if check(QueryAnswer, i, matching(DB, Query(:,:,i))); % Change here to test.
        count = count + 1;
    end
end
acc = count / length(D) * 100;
sprintf('Accuracy is %f%%',acc)