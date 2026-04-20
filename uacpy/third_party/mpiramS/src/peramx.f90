program peramx
!
! RAM PE example - now in Fortran 95!
!
! This code was copied wholesale from Matt Dzieciuch's (@SIO) matlab version, and 
! adapted to Fortran 95.  Matt, in turn, developed his code based on Mike Collins' 
! original FORTRAN version.
!
! B. Dushaw  May 2008
! Single precision version July 2013

use kinds
use interpolators
use envdata
use param
use fld

implicit none

real(kind=wp) :: cmin
real(kind=wp),dimension(:),allocatable :: rp0

real(kind=wp),dimension(:),allocatable :: zg1
complex(kind=wp),dimension(:),allocatable :: psi1
complex(kind=wp),dimension(:,:,:),allocatable :: psif  ! (nzo, nf, nr)

! input parameters - c.f., file "in.pe"
integer :: dzm, iflat, ihorz, ibot
real(kind=wp) :: fc,Q,T,dum
real(kind=wp),dimension(:),allocatable :: zsrc,rmax
character(len=256) :: name1,name2,name3,name4  ! ssp, bathymetry, ranges, sediment filenames
real(kind=wp),dimension(:),allocatable :: eps
real(kind=wp),dimension(:,:),allocatable :: cq

integer :: nss

integer :: nb,nzp,nrp,nrp0,n,nf1,nf,nr
real(kind=wp) :: bw, fs, Nsam, df, tmp
real(kind=wp),dimension(:),allocatable :: frq

integer ::  nzo,icount
real(kind=wp) :: omega

real(kind=wp) :: rate
integer :: t1,t2,cr,cm

integer :: ii,jj,iff,ir,length

integer, parameter :: nunit=2
complex(kind=wp), parameter :: j=cmplx(0.0_wp,1.0_wp,wp)
complex(kind=wp) :: scl

interface
  subroutine ram(zsrc,rg)

    use kinds
    use interpolators
    use envdata
    use fld
    use param

    implicit none

    real(kind=wp),dimension(:),intent(in) :: zsrc,rg

  end subroutine ram

end interface

allocate(zsrc(1))

open(nunit,file='in.pe',status='old')
read (nunit,*) dum                    ! skip the first line (comment/title)
read (nunit,*) fc, Q                  ! center frequency (Hz) and Q value
read (nunit,*) T                      ! time window width (s)
read (nunit,*) zsrc(1)                ! source depth (m)
read (nunit,*) deltaz                 ! depth accuracy parameter (m)
read (nunit,*) deltar                 ! range accuracy parameter (m)
read (nunit,*) np, nss                ! np-# pade coefficients, ns-# stability terms
read (nunit,*) rs                     ! stability range (m)
read (nunit,*) dzm                    ! output depth decimation (integer)
read (nunit,'(a)') name1              ! sound speed filename
name1=trim(adjustl(name1))
read (nunit,*) iflat                  ! 0=no flat earth transform, 1=yes
read (nunit,*) ihorz                  ! 0=no horizontal linear interpolation, 1=yes
read (nunit,*) ibot                   ! 0=no bottom, 1=bottom and read a file
read (nunit,'(a)') name2              ! bathymetry filename
name2=trim(adjustl(name2))
read (nunit,'(a)') name3              ! output ranges filename
name3=trim(adjustl(name3))

! Read bottom properties (sedlayer, nzs, cs, rho, attn)
read (nunit,*) sedlayer
read (nunit,*) nzs
read (nunit,*) isedrd

if (isedrd==1) then
   ! Range-dependent sediment: read filename and load profiles
   read (nunit,'(a)') name4
   name4=trim(adjustl(name4))

   ! Temporary defaults (overridden by file)
   nrp_sed=1
   allocate(cs(nzs,1),rho(nzs,1),attn(nzs,1))
   cs(:,1)  = 0.0_wp
   rho(:,1) = 1.2_wp
   attn(:,1)= 0.5_wp

   close(nunit)

   ! Read sediment profile file (same format as SSP: "-1 range_km" headers)
   print *,'Reading sediment file: ', trim(name4)
   open(nunit,file=name4,status='old')

   ! First pass: count profiles
   deallocate(cs,rho,attn)
   nrp_sed=0
   do
      read(nunit,*,end=4) dum
      if (dum<0) nrp_sed=nrp_sed+1
   end do
4  print *,'Found ',nrp_sed,' sediment profiles.'
   rewind(nunit)

   allocate(rp_sed(nrp_sed), cs(nzs,nrp_sed), rho(nzs,nrp_sed), attn(nzs,nrp_sed))

   ! Second pass: read profiles (nzs values per line)
   do ii=1,nrp_sed
      read(nunit,*) dum, rp_sed(ii)
      rp_sed(ii) = rp_sed(ii)*1000.0_wp   ! convert km to m
      read(nunit,*) (cs(jj,ii), jj=1,nzs)
      read(nunit,*) (rho(jj,ii), jj=1,nzs)
      read(nunit,*) (attn(jj,ii), jj=1,nzs)
   end do
   close(nunit)

else
   ! Range-independent sediment: read nzs-element arrays from in.pe
   nrp_sed=1
   allocate(cs(nzs,1),rho(nzs,1),attn(nzs,1))
   read (nunit,*) (cs(jj,1), jj=1,nzs)
   read (nunit,*) (rho(jj,1), jj=1,nzs)
   read (nunit,*) (attn(jj,1), jj=1,nzs)
   close(nunit)
end if

! Read output ranges from file
print *,'Reading output ranges file: ', trim(name3)
open(nunit,file=name3,status='old')
nr=0
do
   read(nunit,*,end=6) dum
   nr=nr+1
end do
6 print *,'Found ',nr,' output ranges.'
rewind(nunit)
allocate(rmax(nr))
do ii=1,nr
   read(nunit,*) rmax(ii)
end do
close(nunit)

print *
print '(a)','INPUT PARAMETERS:'
print '(a,f10.2)','Center frequency (Hz): ', fc
print '(a,f4.1)','Q: ', Q
print '(a,f5.2)','Bandwidth (f0/Q - Hz): ', fc/Q
print '(a,f6.1)','Time window width (s): ', T
print '(a,f8.1)','Source depth (m): ', zsrc(1)
print '(a,i6)','Number of output ranges: ', nr
print '(a,f12.1)','First range (m): ', rmax(1)
print '(a,f12.1)','Last range (m): ', rmax(nr)
print '(a,f5.2)','Depth accuracy (deltaz, m): ', deltaz
print '(a,f6.2)','Range accuracy (deltar, m): ', deltar
print '(a,i2)','No. of pade coefficients: ', np
print '(a,i2)','No. of stability terms: ', nss
print '(a,f7.1)','Stability range (m): ', rs
print '(a,i4)','Output depth decimation: ', dzm
print '(a,a)','Sound speed filename: ', trim(name1)
print '(a,i1)','Flat-earth transform flag: ', iflat
print '(a,i1)','Horizontal interpolation flag: ', ihorz
print '(a,i1)','Ocean bottom flag: ', ibot
print '(a,a)','Ocean bottom filename: ', trim(name2)
print '(a,a)','Ranges filename: ', trim(name3)
print '(a,f8.1)','Sediment layer (m): ', sedlayer
print '(a,i4)','Sediment depth points (nzs): ', nzs
if (isedrd==1) then
   print '(a,i4,a)','Sediment: range-dependent (',nrp_sed,' profiles)'
else
   print '(a,*(f8.2))','Sediment speed (cs): ', cs(:,1)
   print '(a,*(f8.3))','Sediment density (rho): ', rho(:,1)
   print '(a,*(f8.3))','Sediment attenuation: ', attn(:,1)
end if
print *

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! SOUND SPEED SET UP
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (name1=='munk') then
   !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   ! *** set up range independent Munk SSP ***
   nrp=1
   allocate(rp(nrp))   ! must be a vector, since ram expects a vector
   rp(nrp)=0.0_wp

   nzp=5001
   allocate(zw(nzp),cw(nzp,nrp))
   forall (ii=1:nzp) zw(ii)=real(ii-1,wp)
   cw(:,1)=cssprofile(zw,0.0_wp)     ! 2nd parameter is latitude
else
   ! open and read the sound speed from file name1
   print *,'Reading sound speed file.'
   open(nunit,file=name1,status='old')
   ! first run through and count the -1's to get the number of profiles and depths
     nzp=0
     nrp0=0
     do
        read(nunit,*,end=10) dum
        if (dum<0) then
            nrp0=nrp0+1 
        end if
        if ((dum>=0).and.(nrp0==1)) nzp=nzp+1
     end do
 10  print *,'Found ',nrp0,' profiles, and ',nzp,' depths.' 
   close(nunit)
     allocate(rp0(nrp0),zw(nzp),cq(nzp,nrp0))
   open(nunit,file=name1,status='old')
     do ii=1,nrp0
        read(nunit,*) dum,rp0(ii)
        rp0(ii)=rp0(ii)*1000.0_wp
        do jj=1,nzp
            read(nunit,*) zw(jj),cq(jj,ii)
        enddo
     end do 
   close(nunit)

   !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   ! Sound speeds are read in. 
   ! Now interpolate in range to 10-km intervals (linear), say, if requested.
   
   if (ihorz==1) then
   ! Horizontal
     nrp=nint(maxval(rmax)/10000.0_wp)
     call linspace(rp, rp0(1),maxval(rmax),nrp)
     allocate(cw(nzp,nrp))
     do jj=1,nzp
        cw(jj,:)=interp1(rp0,cq(jj,:),rp,cq(jj,1))
     enddo
   else
     nrp=nrp0
     allocate(rp(nrp),cw(nzp,nrp))
     rp=rp0
     cw=cq
   end if
   deallocate(cq,rp0)

     ! flat earth transformation
   if (iflat==1) then
      print *
      print *,'Applying flat-earth transformation.'
      print *
      allocate(eps(nzp))
      eps=zw*invRe
      zw=zw*(1.0_wp+(1.0_wp/2.0_wp)*eps+(1.0_wp/3.0_wp)*eps*eps)
      forall(ii=1:nrp) cw(:,ii)=cw(:,ii)*(1+eps+eps*eps)
      deallocate(eps)
      ! also transform the source depth
      zsrc(1)=zsrc(1)*(1.0_wp+(1.0_wp/2.0_wp)*zsrc(1)*invRe +   &
            (1.0_wp/3.0_wp)*(zsrc(1)*invRe)**2)
   end if

endif
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

! mean sound speed
n=size(cw)
c0=sum(cw)/n
ic0=1.0_wp/c0
cmin=minval(cw)     ! minimum sound speed for calculating tdelay
print '(a,f10.2,a,f10.2)', 'c0=',c0,' cmin=',cmin

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! BOTTOM SET UP
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (ibot==0) then
   ! use default flat bottom
   nb=1
   allocate(rb(nb),zb(nb))
   rb(nb)=0.0_wp
   zb(nb)=maxval(zw) - 400.0_wp    
    ! Set the ocean floor to be 400 m shallower than deepest sound speed depth.
    ! c.f., attenuation below
else
   ! open and read in the bathymetry from file name2
   print *,'Reading bathymetry file.'
   open(nunit,file=name2,status='old')
! first find the number of values
   nb=0
   do
     read(nunit,*,end=15) dum
     nb=nb+1 
   end do
15 print *,'Found ',nb,' bathymetry values.' 
   rewind(nunit)
   allocate(rb(nb),zb(nb))
   do ii=1,nb
      read(nunit,*) rb(ii),zb(ii)
   end do 
   close(nunit)
endif

   ! flat earth transformation
if (iflat==1) then
   allocate(eps(nb))
   eps=zb*invRe
   zb=zb*(1.0_wp+(1.0_wp/2.0_wp)*eps+(1.0_wp/3.0_wp)*eps*eps)
   deallocate(eps)
   !do ii=1,nb
   !   if (zb(ii)>maxval(zw)) then
   !      print *,'Bottom deeper than water sound speed',zb(ii),maxval(zw)
   !      zb(ii)=maxval(zw)
   !   end if
   !end do  
end if

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! BOTTOM PROPERTIES MODEL
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! Bottom properties (sedlayer, nzs, cs, rho, attn) were read from in.pe above.
! cs is sediment speed relative to the water sound speed (nzs values:
!   at surface, at seafloor, evenly spaced through sediment, at domain bottom).
! rho is sediment density (g/cm^3), nzs values.
! attn is sediment attenuation (dB/wavelength), nzs values.

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!                Example using fc=75 Hz, Q=2, and time window T=2.0 
bw=fc/Q        ! 75/2=37.5 Hz bandwidth
fs=4.0_wp*fc   ! 4*75 Hz = 300 Hz sampling frequency
!dt=1.0_wp/fs   ! 1/300 s sampling interval
Nsam=fs*T      ! 300 Hz*2 s = 600 samples
df=fs/Nsam     ! 300/600 = 0.5 Hz frequency interval

! frequency set up
tmp=(bw-df)/df
nf1=int(tmp) + 1  ! (37.5-0.5)/0.5 + 1 = 75 frequencies, one half
nf=2*nf1+1          ! including 0, 151 frequencies altogether.
allocate(frq(nf))
do ii=1,nf1
  frq(ii)=-(nf1-(ii-1))*df + fc
enddo
frq(nf1+1)= fc
do ii=1,nf1
  frq(ii+nf1+1)=ii*df + fc
enddo

! frq=[df:df:bw];
! frq=[-fliplr(frq) 0 frq]+fc;
! nf=size(frq)

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

zmax=maxval(zw)

! icount is number of depths in zg, psi
icount=floor(zmax/deltaz-0.5_wp)+2

nzo=0
do ii=1,icount,dzm
   nzo=nzo+1 
end do
! nzo is number of depths in psi1, psif, zg1

allocate(psif(nzo,nf,nr))

! Pre-allocate zg before the parallel region to avoid an OpenMP race
! condition: without this, multiple threads entering ram() could
! simultaneously see zg as unallocated and both try to allocate it.
call linspace(zg, 0.0_wp, zmax, icount)

call system_clock(count_rate=cr)
call system_clock(count_max=cm)
rate=real(cr)

print *,nf,' total frequencies, ',nr,' output ranges'
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! Meat and Potatoes
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!$OMP PARALLEL PRIVATE (psi1,omega,scl,t1,t2,cr,rate,ir)
allocate(psi1(nzo))
!$OMP DO SCHEDULE(STATIC,1)
  do iff=1,nf
    call system_clock(t1)

    frqq=frq(iff)
    ns=nss

    call ram(zsrc,rmax)

    omega=2.0_wp*pi*frqq
    do ir=1,nr
      psi1=psi(1:icount:dzm,ir)
      ! 3-D scaling
      scl=exp(j*(omega/c0*rout(ir) + pi/4.0_wp))/4.0_wp/pi
      psif(:,iff,ir)=scl*psi1
    end do

    call system_clock(t2,cr)
    rate=real(cr)
    print '(a,i4,1x,f8.3,a,f9.3,a)','iff,frqq = ', iff,frqq, &
            '  Time: ',nint(100.0_wp*dble(t2-t1)/rate)/100.0_wp,' sec'
  end do
!$OMP END DO
!$OMP END PARALLEL

  allocate(zg1(nzo))
  zg1=zg(1:icount:dzm)
!  Remove the flat-earth transform (or most of it, anyways)
  if (iflat==1) then
    allocate(eps(nzo))
    eps=zg1*invRe
    zg1=zg1/(1.0_wp+(1.0_wp/2.0_wp)*eps+(1.0_wp/3.0_wp)*eps*eps)
    deallocate(eps)
  end if

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! Output: write psif.dat with all ranges
! Format:
!   Record 1: Nsam, nf, nzo, nr, c0, cmin, fs, Q  (8 reals)
!   Record 2: frq(1:nf)                            (nf reals)
!   Record 3: rout(1:nr)                            (nr reals)
!   Records 4..3+nzo*nr: for each range ir, for each depth ii:
!     zg1(ii), re(psif(ii,1,ir)), im(psif(ii,1,ir)), ..., re(psif(ii,nf,ir)), im(psif(ii,nf,ir))
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

! Record length in iolength units (same units as recl= expects).
! Must be large enough for ALL record types:
!   header: 8 reals;  frq: nf reals;  rout: nr reals;  depth: 1+2*nf reals
! Use inquire to get the iolength of a single real, then multiply.
block
  integer :: rl1
  inquire(iolength=rl1) fc        ! iolength of one real(wp)
  length = max(8, nf, nr, 1+2*nf) * rl1
end block

open(nunit, form='formatted',file='recl.dat')
write(nunit,*) length
close(nunit)

open(nunit, access='direct',recl=length,file='psif.dat')

! Record 1: header parameters
write(nunit,rec=1) Nsam,real(nf,wp),real(nzo,wp),real(nr,wp),c0,cmin,fs,Q
! Record 2: frequency vector
write(nunit,rec=2) frq
! Record 3: output ranges
write(nunit,rec=3) rout

! Records 4+: data blocks, one per range, each containing nzo depth records
do ir=1,nr
  do ii=1,nzo
    write(nunit,rec=3+(ir-1)*nzo+ii) zg1(ii), &
        ((real(psif(ii,jj,ir))),(aimag(psif(ii,jj,ir))),jj=1,nf)
  end do
end do

close(nunit)

print *, 'ALL DONE! '
print '(a,i6,a,i4,a)','Wrote ',nzo,' depths x ',nr,' ranges to psif.dat'
print *,'recl.dat has the record length of the direct access file.'

stop

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CONTAINS
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function cssprofile(z,lat)
! canonical sound speed profile
!
! z (meters)
! c (meters/second)
!
! if abs(lat)<67.5 temperate ocean (default)
! else polar ocean

use kinds

implicit none

real(kind=wp),dimension(:),allocatable :: cssprofile

real(kind=wp) :: lat
real(kind=wp),dimension(:) :: z 
real(kind=wp),dimension(:),allocatable :: eta

integer :: i,n
real(kind=wp) :: gammaa, za, h, ca

gammaa=0.0113_wp/1000.0_wp
za=1000.0_wp
h=1000.0_wp
ca=1500.0_wp

n=size(z)
allocate(cssprofile(n),eta(n))

if (abs(lat)<67.5_wp) then
  !temperate ocean
  do i=1,n
    eta(i)=2.0_wp*(za-z(i))/h
    cssprofile(i)=ca*(1.0_wp+h*gammaa*(exp(eta(i)) - eta(i) - 1.0_wp)/2.0_wp)
  end do
else
  !polar ocean
  do i=1,n
    cssprofile(i)=ca*(1+gammaa*z(i))
  end do
end if

end function cssprofile

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end program peramx

