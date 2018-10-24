clear all;
D=load('./USPS/trai_data.txt');, trai_label=load('./USPS/trai_label.txt');
Q=load('./USPS/test_data.txt');, test_label=load('./USPS/test_label.txt');
[Dim,trai_num]=size(D);,[Dim,test_num]=size(Q);
for i = 1 : trai_num, D(:,i)=D(:,i)./norm(D(:,i));, end
for i = 1 : test_num, Q(:,i)=Q(:,i)./norm(Q(:,i));, end
W=zeros(Dim,100,10);
figure(1), clf, axis square;
for j = 0 : 9
  X=D(:,find(trai_label==j));
  C=X*transpose(X);
  [eig_vec, eig_val]=eig(C);
  [value index]=sort(-diag(eig_val));
  W(:,:,j+1)=eig_vec(:,index(1:100));
  figure(1),subplot(2,5,j+1), plot(-value(1:100));
  s=sprintf('class %d', j);, title(s);
  xlabel('dim');,ylabel('eigenvalue');
  fprintf('class %d ... OK\n',j);
end
figure(2), clf, axis square;
IMG=zeros(16,16);
count=1;
for j = 0 : 9
  for k = 1 : 10
    for i = 1 : 16, IMG(i,:)=transpose(W((i-1).*16+1:i.*16,k,j+1));,end
    IMG=IMG-min(min(IMG));, IMG=IMG./max(max(IMG));
    figure(2),subplot(10,10,count),imshow(IMG);
    count=count+1;
  end
end
