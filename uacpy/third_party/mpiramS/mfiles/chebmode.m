function [cp, cg, evf]=chebmode( zi, ci, frq, nm, zr)
% function [cp, cg, evf]=chebmode( zi, ci, frq, nm, zr)
% acoustic mode awcg driver
%

if nargin==4
  zr=[];
end

nf=length(frq);
nz=length(zr);

if size(ci,1)==1
  ci=ci(:);
end

if nargout>2
  ci=ci(:,1);
  if nz>1
    evf=zeros(nz,nm,nf);
  else
    evf=zeros(nm,nf);
  end
end

[nd, nr]=size(ci);

cp=zeros(nr,nm,nf);
cg=zeros(nr,nm,nf);

ncby=floor(1.5*nm);
ncut=floor(ncby/2);

% adjust decaytarget to be on the lower flat part of the chebyshev
% spectrum plotted below.
decaylimit=-9.0;
decaytarget=-13.0;
dddn=-15/200;

for ir=1:nr
  ss2=1./(ci(:,ir).^2);

  for ifm=1:nf
    fc=frq(ifm);

    while 1
      fprintf(1,'ir: %d frq: %.0f ncby: %d ', ir, fc, ncby);
      disp(' ')

      [M, H]=chebfit( zi, ss2, ncby);
      M(ncut+1:ncby)=0;
      if nargout>2
        [icp, icg, v, chsp] = awcg( M, H, fc, nm, zr);
        if nz>1
          evf(:,:,ifm)=v;
        else
          evf(:,ifm)=v.';
        end
      else
        [icp, icg, v, chsp] = awcg( M, H, fc, nm);
      end

      %plot the mode chebyshev spectrum
      %clf
      %plot( log10(chsp(:,[1:5:nm])))
      %ylabel('Chebyshev coefficient')
      %grid on; zoom on

      decay=sum(chsp(ncby-12:ncby-2,:))./sum(chsp(1:10,:));
      decay=log10(max(decay));
%      fprintf(1,'decay: %.2f\n', decay);

      if( decay>decaytarget)
        delcheby=ceil( -0.5*(decay-decaytarget)/dddn);
      else
        delcheby=floor( -0.5*(decay-decaytarget)/dddn);
      end

      if( ~delcheby) delcheby=10; end
      if( decay>decaylimit) delcheby=max([ 50 delcheby]); end
      ncby=ncby+delcheby;
      if (ncby<nm)
        ncby=floor(1.5*nm);
      end
      ncut=floor(ncby/2);

      if (decay<decaylimit) break; end
    end

    cp(ir,:,ifm)=icp;
    cg(ir,:,ifm)=icg;
  end
  clear ifm icp icg fc

  cp=squeeze(cp);
  cg=squeeze(cg);

  if nargout>2
    evf=squeeze(evf);
  end

end

return

