r=30;
x=2;

if exist('W')==0
  D=load('./USPS/trai_data.txt');, trai_label=load('./USPS/trai_label.txt');
  Q=load('./USPS/test_data.txt');, test_label=load('./USPS/test_label.txt');
  [Dim,trai_num]=size(D);,[Dim,test_num]=size(Q);
  for i = 1 : trai_num, D(:,i)=D(:,i)./norm(D(:,i));, end
  for i = 1 : test_num, Q(:,i)=Q(:,i)./norm(Q(:,i));, end
  W=zeros(Dim,100,10);
  for j = 0 : 9
    X=D(:,find(trai_label==j));
    C=X*transpose(X);
    [eig_vec, eig_val]=eig(C);
    [value index]=sort(-diag(eig_val));
    W(:,:,j+1)=eig_vec(:,index(1:100));
    fprintf('class %d ... OK\n',j);
  end
end
figure(1), clf, hold on;
IMG=zeros(16,16);
for i = 1 : 16, IMG(i,:)=transpose(Q((i-1).*16+1:i.*16,x));,end
IMG=IMG-min(min(IMG));, IMG=IMG./max(max(IMG));
figure(1),subplot(2,10,5),imshow(IMG);
title('test sample');
for j = 0 : 9
  c=tranpose(W(:,1:r,j+1))*Q(:,x);
  QA=W(:,1:r,j+1)*c;
  for i = 1 : 16, IMG(i,:)=transpose(QA((i-1).*16+1:i.*16,1));,end
  IMG=IMG-min(min(IMG));, IMG=IMG./max(max(IMG));
  figure(1),subplot(2,10,11+j),imshow(IMG);,
  s=sprintf('class %d',j);, title(s);
end
fprintf('r=%d\n',r);