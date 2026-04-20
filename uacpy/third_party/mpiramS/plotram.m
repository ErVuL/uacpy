clear all
close all

%filename='/home/dushaw/RAM/psif.dat'; 
filename='psif.dat'; 

% Record length of direct access file.  
% The file with this number is written by mpipe
load recl.dat
% If the record length of the direct access file is 2424, then matlab 
% should read 303 real*8 elements to read a complete record.

% This script is coded for the Sun compiler output.

% For intel compiler:  
%  (guess what?  Every compiler is different in how data are stored??!!??)
%recl=recl*4;

fid = fopen(filename, 'r','ieee-le');

% First read a few parameters
offset=0*recl;
status = fseek(fid, offset, 'bof');
q=fread(fid,8,'real*4');

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
q=fread(fid,nf,'real*4');
frq=q(1:nf)';

% Now read the depth and psif rows.
psif=zeros(nzo,nf);
zg=zeros(1,nzo);
isz=1+2*nf;   % The number of data to read.
              % One for depth, and nf real and nf imaginary values (psif) 
for ii=1:nzo
   offset=(ii+1)*recl;
   status = fseek(fid, offset, 'bof');
   q=fread(fid,isz,'real*4');
   zg(ii)=q(1);
   psif(ii,:)=q(2:2:end)+i*q(3:2:end);  
end

fclose(fid);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Should have all we need for plotting now!

% Requires that psif and other variables be read in.
% Proceeding with the remainder of the script.

path('./mfiles',path);      % We need a couple of routines from this directory

nyqst=(nf+1)/2;
fc=frq(nyqst);
bw=fc/Q;
%wind=sinc((frq-fc)/bw);
wind=hann((frq-fc),2*bw);
%wind=ones(size(frq));
%wind=zeros(size(frq)); wind(nyqst)=1;

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ptz=zeros(nzo,N);
for iz=1:nzo
  data=wind.*conj(psif(iz,:)).*exp(i*2*pi*frq*tdelay(nro));
  data=[ data(nyqst:nf), zeros(1,N-nf), data(1:nyqst-1)];
  ptz(iz,:)=ifft( data);
end
clear iz data

if 1==2,
%g=gpuDevice(1);
ptzg=gpuArray(zeros(nzo,N));
psifg=gpuArray(psif);
tdelayg=gpuArray(tdelay);
windg=gpuArray(wind);

for iz=1:nzo
  data=windg.*conj(psifg(iz,:)).*exp(i*2*pi*frq*tdelayg(nro));
  data=[ data(nyqst:nf), zeros(1,N-nf), data(1:nyqst-1)];
  ptzg(iz,:)=ifft(data);
end
ptz=gather(ptzg);
clear iz data tmp ptzg psifg tdelayg windg
%reset(g)

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
threshold=-36;

figure(1);
cti=6; nsbi=6;
ncol=-nsbi*threshold/cti;
ntv=-threshold/ncol;
scale=[threshold+cti:(0-threshold)/(ncol):0]-ntv/2;
cmap=clrscl('wcbmry',ncol);
colormap(cmap);
clear nsbi ntv ncol

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pkdm=max(abs(ptz(:)));
data=20*log10( abs(ptz)/pkdm);
ind=find(data==-inf);
data(ind)=threshold;
clear ind pkdm

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clf
imagesc( scale, [0 1]', [scale' scale']', [threshold 0.0])
set( gca, 'position',  [0.1 0.1 0.2 0.02])
hold on
contour( scale, [0 1]', [scale' scale']', [threshold:cti:0], 'k')
hold off
set( gca, 'xtick', [threshold+cti 0]);
set( gca, 'ytick', []);
axis([ threshold+cti 0 0 1])
xlabel('Power (dB)');

axes;
set( gca, 'position',  [0.1 0.2 0.8 0.65])
imagesc( taxis, zg, data, [threshold 0.0]);
xlabel('travel time (s)');
ylabel('z (m)');
stt=sprintf('Split Step Pade PE intensity, rg=%5.0f km', rout(nro)*1e-3);
title( stt)
grid on; zoom on
set( gca, 'xlim', [tmin tmax]);
set( gca, 'ylim', [zmin zmax]);
drawnow;

