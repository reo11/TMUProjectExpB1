c = 20;
n = 10;

path = '../dataset/DB/jpeg/'

for i=1:c
	for j=1:n
		str = strcat(path,num2str(n*(i-1)+j-1, '%03d'),'.jpg');
		img = imread(str);
		DB(:,:,n*(i-1)+j) = img;
	end
end
