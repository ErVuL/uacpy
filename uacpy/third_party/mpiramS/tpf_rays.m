% zzr, czr - depths and speeds of sound speed profile at the VLA. This should be
% a finely sampled profile, say 1m spacing. You have to do the interpolation yourself.

% zaxr, caxr - depth and speed of sound channel axis.

% Be sure to define zbfr - midpoint depth of VLA. 
% This should be the depth of rcvr for your eigenray calculations.

% finala, ray trace arrival angle
% finalt, ray trace travel time
% finalz, ray trace final depth
%finalc, sound speed at ray trace finalz (used in snells's law)

% get ray information
matray2

% read sound speed field
a='test.ssp';
c_ann=load(a);
I=c_ann(:,1)==-1.;
I=find(I);
N1=I(2)-1;
z=c_ann(2:N1,1);
z=-z/1000;
c=c_ann(:,2);
N2=length(c)/N1
c=reshape(c,N1,N2);
r=c(1,:);
c(1,:)=[];

ce=c(:,end);
re=r(end);
ze=z;

zzr=(0:5000)';
czr=interp1(ze,ce,-zzr/1000,'cubic');

figure(2)
plot(czr,-zzr/1000)
hold on

I=find(czr==min(czr));
I=I(1);
zaxr=zzr(I)
caxr=czr(I)
plot(caxr,-zaxr/1000,'or')

figure(1)

%turning-point filter
 tpfa=180*acos(caxr*cos(pi*ra/180)./crec)/pi;
 tpfa=tpfa.*ra./abs(ra);
 tpft=zeros(size(tpfa));
 for ir=1:Nray
   sv2=1./czr.^2 - (cos(pi*tpfa(ir)/180)/caxr).^2;
   sv=zeros(size(sv2));
   ind=find( sv2>=0.0);
   sv(ind)=sqrt(sv2(ind));
   if( tpfa(ir)<0.0)  sv=-sv;  end
   tau=cumtrapz(zzr,sv);
   td0=interp1(zzr,tau,zbfr,'linear');
   tdz=interp1(zzr,tau,-zr(ir)*1000,'linear')-td0;
   tpft(ir)=tt(ir)+tdz;
 end
 clear ir sv2 sv ind tau td0 tdz
 %fprintf(1,'tpft: %10.5f tpfa: %9.5f\n', [tpft; tpfa]);

% clf
 ind=find(sa>0);
 p=plot( tpft(ind), tpfa(ind), 'ko');
 set(p,'MarkerSize',6,'MarkerFaceColor','k')
 set(p,'MarkerSize',6,'MarkerEdgeColor','k')

 hold on; grid on; zoom on;
 ind=find(sa<0);
 p=plot( tpft(ind), tpfa(ind), 'go');
 set(p,'MarkerSize',6,'MarkerFaceColor','g')
 set(p,'MarkerSize',6,'MarkerEdgeColor','g')

 set(gca,'FontSize',12)
 xlabel('Travel-Time (s)','FontSize',14);
 ylabel('Axial Arrival-Angle (\circ)','FontSize',14);
 title('Turning-Point Filter: Eigenray Test Case','FontSize',16)
% clear ind tpft tpfa


