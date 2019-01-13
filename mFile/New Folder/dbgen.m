faceClass = 20;
number = 10;

path = '../facedata/DB/jpeg/'

for i=1:faceClass
	for j=1:number
		str = strcat(path,num2str(number*(i-1)+j-1, '%03d'),'.jpg');
		img = imread(str);
		DB(:,:,number*(i-1)+j) = img;
	end
end
