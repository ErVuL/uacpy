clear all
close all

% This flag might be used (someday) to plot the mode results on top of the PE TPF results.
modeflag=0;

filename='psif.dat';

% Record length of direct access file.
% The file with this number is written by mpipe
load recl.dat
% If the record length of the direct access file is 2424, then matlab
% should read 303 real*8 elements to read a complete record.

%  Guess what?  Every compiler stores data a little differently!

fid = fopen(filename, 'r','ieee-le');

% First read a few parameters
offset=0*recl;
status = fseek(fid, offset, 'bof');
q=fread(fid,8,'real*8');

N=q(1)
nf=q(2)
nzo=q(3)
rout=q(4)
c0=q(5)
cmin=q(6)
fs=q(7)
Q=q(8)

% Then read the frequencies
offset=1*recl;
status = fseek(fid, offset, 'bof');
q=fread(fid,nf,'real*8');
frq=q(1:nf)';

% Now read the depth and psif rows.
psif=zeros(nzo,nf);
zg=zeros(1,nzo);
isz=1+2*nf;   % The number of data to read.
              % One for depth, and nf real and nf imaginary values (psif)
for ii=1:nzo
   offset=(ii+1)*recl;
   status = fseek(fid, offset, 'bof');
   q=fread(fid,isz,'real*8');
   zg(ii)=q(1);
   psif(ii,:)=q(2:2:end)+i*q(3:2:end);
end

fclose(fid);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Requires that psif and other variables be read in.
% Proceeding with the remainder of the script.

path('./mfiles',path);      % We need a several routines from this directory

nyqst=(nf+1)/2;
fc=frq(nyqst);
bw=fc/Q;
wind=sinc((frq-fc)/bw);

% Set tdelay to plot the output in the right time bin.  
% This can be a bit tricky; when in doubt do a ray trace to double check.
% tdelay=rout/c0 - 1.5 ;
T=N/fs;
%tdelay=rout/cmin-0.5*T;

% rout/cmin should be roughly the arrival time of the on axis energy,
% which is generally the latest arriving energy.  So the start time
% of the arrival should be one period T earlier than that time, in theory.
tdelay=rout/(cmin*1.000) - 1.0*T -0.5;
%tdelay=rout/c0 ;

nro=length(rout);

zmin=min(zg); zmax=max(zg);

taxis=tdelay(nro)+[0:N-1]/fs;
tmin=min(taxis); tmax=max(taxis);

zrcv=zg';

ptz=zeros(nzo,N);
for iz=1:nzo
  data=wind.*conj(psif(iz,:)).*exp(i*2*pi*frq*tdelay(nro));
  data=[ data(nyqst:nf), zeros(1,N-nf), data(1:nyqst-1)];
  ptz(iz,:)=ifft( data);
end
clear iz data

% Sound speed at the receiver range is used to construct the local TPF.
% i.e., the local ray angles as they cross the sound channel axis depend 
% on local sound speed.  
% Also returns the depth and sound speed of the sound channel axis
loadc

if modeflag==1,
  thetacut=16;
  nm=50;
  disp('Stand by for mode calculations...')
% Just need the mode phase speeds.
  z0=1000;
  [cp, cg, evfr]=chebmode( zi, ci, frq, nm, z0);
  disp('Done with modes.')
  thm=180*acos(cax./cp)/pi;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clip out an appropriate section of the time front to simulate reception by a VLA.
%ind=find( zrcv>=2500 & zrcv<=3500);
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
%contour( taxis, zrcv, data, [-30:3:0], 'k')
set( gca, 'FontSize',12)
xlabel('\tau (s)','FontSize',14);
ylabel('z (m)','FontSize',14);
grid on
zoom on
drawnow
disp('Hit return to continue')
pause
disp('O.K. stop hitting return')

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

if modeflag==1,
ind=find( thm>thetacut);
xthm=thm; xthm(ind)=nan;
xtaum=taum; xtaum(ind)=nan;
ind=find( angd>thetacut);
xangd=angd; xangd(ind)=nan;
xtaud=taud; xtaud(ind)=nan;
end

clf
imagesc( taxis, sangle, data, [-30 0]);
hold on
%contour( taxis, sangle, data, [-30:3:0], 'k')
%plot( xtaum, xthm, 'k')

if modeflag==1,
plot( tauw, angw, 'r');
plot( tauw, -angw, 'r');
plot( taur, angr, 'ro')
plot( taur, -angr, 'ro')
plot( xtaud, xangd, 'kx')
plot( xtaud, -xangd, 'kx')
end
set( gca, 'ydir', 'normal')
set( gca, 'FontSize',12)
xlabel('\tau_0 (s)','FontSize',14);
ylabel('\theta_0','FontSize',14);
grid on; zoom on

% Set the colormap
threshold=-36;
cti=6; nsbi=6;
ncol=-nsbi*threshold/cti;
ntv=-threshold/ncol;
scale=[threshold+cti:(0-threshold)/(ncol):0]-ntv/2;
cmap=clrscl('wcbmry',ncol);
colormap(cmap);

clear data pktpf

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculate ray.info already

zbfr=mean(zrcv);
zbfr=zax;
tpf_rays



