r=30;

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

S=zeros(10,1);
CONF=zeros(10,10);
tic
for i = 1 : test_num
  for j = 0 : 9, S(j+1)=sum((transpose(W(:,1:r,j+1))*Q(:,i)).^2);, end
  [value index]=max(S);
  CONF(test_label(i)+1,index)=CONF(test_label(i)+1,index)+1;
  fprintf('test data %d\n',i);
end
toc
accuracy=(sum(diag(CONF))./test_num).*100;
fprintf('accuracy=%3.2f\n',accuracy);
