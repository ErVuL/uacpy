C=load('test.ssp');
I=C(:,1)==-1.;
I=find(I);
N1=I(2)-1;
z=C(2:N1,1);
%z=-z/1000;
c=C(:,2);
N2=length(c)/N1
c=reshape(c,N1,N2);
r=c(1,:);
c(1,:)=[]; 

cl=c(:,end);
zi=0:1:5000;
zi=zi';
ci=interp1(z,cl,zi,'cubic');
I=find(ci==min(ci));
cax=ci(I(1));
zax=zi(I(1));


