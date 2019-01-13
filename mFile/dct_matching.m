function [number, rjflag] = dct_matching(DCT_DB,Query,q)
%DCT‚µ‚Ä“ñæŒë·‚Ì‹ß‚¢‚à‚Ì‚ğ’T‚·ƒvƒƒOƒ‰ƒ€
X = Query(:,:,q);
dblX = double(X);
dctX = dct2(dblX);
dctXlow = dctX(1:15,1:15);

for i=1:200
 imgdctlow = DCT_DB(:,:,i);
 D = (dctXlow - imgdctlow).^2;
 distance(i) = sum(sum(D));
end

for i=1:5
[minimum(i), index(i)] = min(distance);
distance(index(i)) = 10000;
end

number = ceil(index(1)/10)-1;

for i=1:5
perfect_index(i) = ceil(index(i)/10)-1;
end
disp(minimum)
disp(perfect_index)

%reject
if minimum(1) > 10000000000
    number = 100;
    rjflag = 1;
else rjflag = 0;
end
end

