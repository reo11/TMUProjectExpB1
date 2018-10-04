function matching(DB, X)
	dblX = double(X);
	for i=1:200
		A = DB(:,:,i);
		dblA = double(A);
		D = (dblX - dblA).^2;
		distance(i) = sum(sum(D));
	end
	[minimum, index] = min(distance);
	number = ceil(index/10);
	sprintf('X is Person %d.',number)
end
