function [ cmap]=clrscl(palette,ncol)
% function [ cmap]=clrscl(palette,ncol)
% generate colorscale
% palette: r,g,b,c,y,m,w,k (full scale colors)
%          C,Y,M           (equal intensity)
%
% ncol: number of output colors
%

q=1/sqrt(2);

cmap=[];
for ic=1:length(palette)
  switch palette(ic);
    case 'r'
      cmap=[cmap; 1 0 0];
    case 'g'
      cmap=[cmap; 0 1 0];
    case 'b'
      cmap=[cmap; 0 0 1];
    case 'C'
      cmap=[cmap; 0 q q];
    case 'c'
      cmap=[cmap; 0 1 1];
    case 'Y'
      cmap=[cmap; q q 0];
    case 'y'
      cmap=[cmap; 1 1 0];
    case 'M'
      cmap=[cmap; q 0 q];
    case 'm'
      cmap=[cmap; 1 0 1];
    case 'w'
      cmap=[cmap; 1 1 1];
    case 'k'
      cmap=[cmap; 0 0 0];
    otherwise
      error('palette error');
  end
end

nmap=size(cmap,1);
if nmap<2
  error('palette error');
end

if nargin<2
  ncol=nmap;
end

ncol=floor(ncol);
if ncol<nmap
  error('palette error');
end

xi=[0:nmap-1]/(nmap-1);
xo=[0:ncol-1]/(ncol-1);

cmap=interp1(xi,cmap,xo,'linear');
cmap=cmap/max(cmap(:));
cmap=abs(cmap);

if nargout>0
  return
end

xbar=[1:ncol];

clf
plot( xbar, cmap(:,1), 'r');
hold on
plot( xbar, cmap(:,1), 'ro');
plot( xbar, cmap(:,2), 'g');
plot( xbar, cmap(:,2), 'gx');
plot( xbar, cmap(:,3), 'b');
plot( xbar, cmap(:,3), 'b^');
hold off
grid on;
set( gca, 'xlim', [1 ncol]);

colormap(cmap);
colorbar

pause

threshold=-30; ntv=-threshold/ncol;

scale=[threshold:(0-threshold)/(ncol):0]-ntv/2;
clf
imagesc( scale, [0 1]', [scale' scale']', [threshold 0]);
set( gca, 'position',  [0.1 0.1 0.8 0.8])
hold on
contour( scale, [0 1]', [scale' scale']', [threshold:3:0], 'k')
hold off
set( gca, 'xtick', [threshold 0]);
set( gca, 'ytick', []);
axis([ threshold 0 0 1])
xlabel('Power (dB)');

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

brt=1.0; sat=1.0;
cmap=[];
for i=0:ncol-1
  hue = 210.0-240.0*i/ncol;
  [red,green,blue]= hsb( hue, sat, brt);
  cmap=[ cmap; [red, green, blue]];
end
cmapc=cmap;
clear brt sat i hue red green blue ncol cmap

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
threshold=-30;

scale=[threshold:(0-threshold)/100.0:0];
clf
imagesc( scale, [0 1]', [scale' scale']', [threshold 0]);
set( gca, 'position',  [0.1 0.1 0.8 0.8])
hold on
contour( scale, [0 1]', [scale' scale']', [threshold:3:0], 'k')
hold off
set( gca, 'xtick', [threshold 0]);
set( gca, 'ytick', []);
axis([ threshold 0 0 1])
xlabel('Power (dB)');
colormap( cmapc);

return

cmapc=[
1.0 1.0 1.0;
1.0 1.0 0.0;
0.0 1.0 0.0;
0.0 0.8 1.0;
1.0 0.0 1.0;
1.0 0.7 0.0;
1.0 0.0 0.0;
];

colormap( cmapc);

cmapc=[
1 1 1;
1 1 0;
0 1 0;
0 1 1;
0 0 1;
1 0 1;
1 0 0;
0 0 0;
];

colormap( cmapc);


