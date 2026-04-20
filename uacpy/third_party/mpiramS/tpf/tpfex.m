%tpfex.m
% turning-point filter example
%
disp('turning-point filter example');

load rimodeex
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ind=find( zrcv>=2500 & zrcv<=3500);
ind=find( zrcv>=500 & zrcv<=1500);
zrcv=zrcv(ind);
ptz=ptz(ind,:);
clear ind
nz=length(zrcv);

pkdm=max(max(abs(ptz)));
data=20*log10( abs(ptz)/pkdm);
clear pkdm

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clf
imagesc( taxis, zrcv, data, [-30 0.0]);
hold on
contour( taxis, zrcv, data, [-30:3:0], 'k')
xlabel('\tau (s)');
ylabel('z (m)');
grid on
zoom on
drawnow
disp('Hit return to continue')
pause

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

omega=2*pi*[0:N/2 1-N/2:-1]'/T;
wc=2*pi*fc;

zbeam=3000;
zbeam=zax;

kax=wc/cax;

da=0.2;
sangle=[-20:da:20];
na=size(sangle,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dmodf=fft(ptz.');

tpf=zeros(na,N);
tpfgn=zeros(na,1);
for ai=1:na
  ang=sangle(ai);
  kh=kax*cos(pi*ang/180);
  kz2=-kh*kh+wc*wc./(ci.*ci);
  kz=sqrt(kz2);
  if( ang<0.0)  kz=-kz;  end
  tau=cumtrapz(zi, kz)/wc;
  tauax=interp1( zi, tau, zbeam, 'linear');
  tdz=(tau-tauax);
  tdz=interp1( zi, tdz, zrcv, 'linear');

  %tdx=bfdist(zeros(nz,1),zeros(nz,1),-(zrcv-zax),0.0,0.0);
  %tdx=tdx/cax;
  %td=(tdz+tdx).';
  td=tdz.';

  phshft=exp(-j*(omega+wc)*real(td));
  phshft=phshft.*exp(-abs((omega+wc)*imag(td)));

  %gn=abs(phshft);
  %tpfgn(ai)=gn(:)'*gn(:);

  tpf(ai,:)=ifft(sum((dmodf.*phshft).'));
end
 
clear ai ang kh kax kz2 kz tau tauax tdz tdx td
clear phshft gn dmodf omega wc

%tpfg=tpfgn/max(tpfgn);
%tpfg=repmat(tpfg, [1 N]);
%clear tpfgn

%pktpf=max(max(tpfg.*abs(tpf)));
pktpf=max(max(abs(tpf)));
warning off
%data=20*log10( tpfg.*abs(tpf)/pktpf);
data=20*log10( abs(tpf)/pktpf);
warning on

ind=find( thm>thetacut);
xthm=thm; xthm(ind)=nan;
xtaum=taum; xtaum(ind)=nan;
ind=find( angd>thetacut);
xangd=angd; xangd(ind)=nan;
xtaud=taud; xtaud(ind)=nan;

clf
imagesc( taxis, sangle, data, [-30 0]);
hold on
contour( taxis, sangle, data, [-30:3:0], 'k')
%plot( xtaum, xthm, 'k')
plot( tauw, angw, 'r');
plot( tauw, -angw, 'r');
plot( taur, angr, 'ro')
plot( taur, -angr, 'ro')
plot( xtaud, xangd, 'kx')
plot( xtaud, -xangd, 'kx')
set( gca, 'ydir', 'normal')
xlabel('\tau_0 (s)');
ylabel('\theta_0');
grid on; zoom on
clear data pktpf

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


