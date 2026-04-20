function [cp, cg, v, chsp] = awcg( M, H, fc, nm, zr)
% function [cp, cg, v, chsp] = awcg( M, H, fc, nm, zr)
% Acoustic mode generation program
% using chebyshev polynomials as basis functions.
%
% fc is the acoustic frequency
% nm is the number of modes to solve for
% zr is the depth of the hydrophones
%
% cp is the mode phase speed
% cg is the mode group speed
% v is the mode shape
% chsp is the chebyshev polynomial spectrum
%

if nargin<4 nm=0; end
if nargin<5 zr=[]; v=[]; end

N=length(M);

%The chebyshev differential operator
D=zeros(N,N);
for ik=2:N
  D(ik-1:-2:1,ik)=D(ik-1:-2:1,ik)+2*(ik-1)*ones(floor(ik/2),1);
end
D(1,:)=D(1,:)/2;
clear ik

%The second derivative operator
D2=D*D;

%The convolution operator
G=zeros(N,N);
for ik=1:N 
  G(ik,1:ik)=G(ik,1:ik)+M(ik:-1:1).';
  G(ik,ik:N)=G(ik,ik:N)+M(1:N+1-ik).';
end
for ik=2:N 
  G(ik,1:N+1-ik)=G(ik,1:N+1-ik)+M(ik:N).';
end
G=G/2;

%The vertical wave equation operator
sc=2*pi*fc; 
L=4*D2/H/H + sc*sc*G;

%add in the boundary conditions,
%pressure release surface and bottom
bc(2,:)=ones(1,N);
bc(1,:)=(-1).^([0:N-1]);
cx=bc(:,N-1:N);
bc1=cx\bc;
LR=L(1:N-2,1:N-2) - L(1:N-2,N-1:N)*bc1(:,1:N-2);

%solve the equation
[ V, k2]=eig(LR);

%extract the eigenvalues
k2=diag(k2);
ind=find( imag(k2)==0);
k2=k2(ind); V=V(:,ind);
ind=find( k2>0);
k2=k2(ind); V=V(:,ind);
[ k2, ind]=sort(k2);
V=V(:,ind);
k2=flipud(k2); V=fliplr(V);
kh=sqrt(k2);
nms=length(k2);

if nm>0 & nm<=nms
  nms=nm;
  kh=kh(1:nm);
end

%compute the mode phase speeds
cp=2*pi*fc./kh;

%integral operator
I=[0:N-2-1];
I=(I.*I - 1);
ind=find(I);
I(ind)=-2./I(ind);
I(2:2:N-2)=zeros(floor((N-2)/2),1);
clear ind

%convolution operator with slowness squared
G=zeros(N-2,N-2);
for ik=1:N-2
  G(ik,1:ik)=G(ik,1:ik)+M(ik:-1:1).';
  G(ik,ik:N-2)=G(ik,ik:N-2)+M(1:N-2+1-ik).';
end
for ik=2:N-2
  G(ik,1:N-2+1-ik)=G(ik,1:N-2+1-ik)+M(ik:N-2).';
end
G=G/2;

if ~isempty( zr)
  %inverse chebyshev operator at mode depths
  zind=find(ones(size(zr))); nzr=max(zind);
  zh=H*[0:10*nms]/10/nms;
  ztind=length(zind)+find(ones(size(zh)));
  zr=[ zr(:); zh(:);]';

  xr=1-2*zr/H;
  clear Ker
  Ker=zeros(N-2,length(zr));
  Ker(1,:)=1;
  Ker(2,:)=xr;
  for ik=3:N-2
    Ker(ik,:)=2.*xr.*Ker(ik-1,:) - Ker(ik-2,:);
  end
  v=zeros(length(zr), nms);
end

cg=zeros(size(cp));
%loop on the number of valid modes
for ip=1:nms

  %convolution operator with mode shape squared
  V2=zeros(N-2,N-2);
  for ik=1:N-2
    V2(ik,1:ik)=V2(ik,1:ik)+V(ik:-1:1,ip).';
    V2(ik,ik:N-2)=V2(ik,ik:N-2)+V(1:N-2+1-ik,ip).';
  end
  for ik=2:N-2
    V2(ik,1:N-2+1-ik)=V2(ik,1:N-2+1-ik)+V(ik:N-2,ip).';
  end
  V2=V2/2;
  V2=V2*V(:,ip);

  % group speed integral
  iv2=H*I*V2/2;
  cg(ip)=iv2/(cp(ip)*H*I*G*V2/2);

  if ~isempty( zr)
    % mode shape inverse transform
    v(:,ip) = Ker.'*V(:,ip)/sqrt(iv2);

    %% make sure the polarity is the same for all modes
    vmax=max(abs(v(ztind,ip))); 
    ind=min(find( abs(v(ztind,ip))>0.1*vmax));
    if v(nzr+ind,ip)<0.0
      v(:,ip)=-v(:,ip);
    end
  end
end

if nm>nms
  cp(nms+1:nm)=0;
  cg(nms+1:nm)=0;
  if ~isempty( zr)
    v(:,nms+1:nm)=0;
  end
end

if ~isempty( zr)
  v=v(zind,:);
end

if nargout==4
  chsp=abs(V(:,1:nms)*diag(k2(1:nms)));
end

return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

