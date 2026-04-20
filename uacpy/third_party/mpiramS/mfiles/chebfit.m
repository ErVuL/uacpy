function [M, H]=chebfit( z, m, ncby)
% function [M, H]=chebfit( z, m, ncby)
% fit data using a chebyshev polynomial
%

  z=z(:); m=m(:);

  H=max(z)-min(z);

  x=1-2*z/H;
  k=0:ncby-1;
  xi=cos( k*pi/(ncby-1));
  mi=interp1(x,m,xi,'spline').';

  mi(1)=mi(1)/2; mi(ncby)=mi(ncby)/2;
  Ker=ones( ncby, length(xi));
  Ker(2,:)=xi;
  for ik=3:ncby
    Ker(ik,:)=2.*xi.*Ker(ik-1,:) - Ker(ik-2,:);
  end
  M=2*Ker*mi/(ncby-1);
  M(1)=M(1)/2; M(ncby)=M(ncby)/2;
return
