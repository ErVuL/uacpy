program peramx
! ram PE example - now in Fortran 95!
! Copied wholesale from Matt Dzieciuch's matlab version, and adapted to Fortran 95.
! Matt, in turn, developed his code based on Mike Collins' old FORTRAN version
!
! B. Dushaw
! Single precision version July 2013

use kinds             ! merely defines the single/double precision kind _wp/_wp2
use interpolators     ! linear and cubic spline interpolators
use envdata           ! the environmental data: sound speeds and parameters
use param             ! a collection of parameters and scalar variables
use fld

use mpi               ! mpi requirement
                      ! This file is the only one that has anything to do with MPI.

implicit none

real(kind=wp) :: cmin
real(kind=wp),dimension(:),allocatable :: rp0

real(kind=wp),dimension(:),allocatable :: zg1
real(kind=wp),dimension(:,:),allocatable :: rpsif,ipsif,rpsiff,ipsiff
complex(kind=wp),dimension(:),allocatable :: psi1
complex(kind=wp),dimension(:,:),allocatable :: psif

! input parameters - c.f., file "in.pe"
integer :: dzm, iflat, ihorz, ibot
real(kind=wp) :: fc,Q,T,dum
real(kind=wp),dimension(:),allocatable :: zsrc,rmax
character(len=20) :: name1,name2     ! sound speed and bathymetry filenames
real(kind=wp),dimension(:),allocatable :: eps
real(kind=wp),dimension(:,:),allocatable :: cq

integer :: nss

integer :: nb,nzp,nrp,nrp0,n,nf1,nf
real(kind=wp) :: bw, fs, Nsam, df
real(kind=wp),dimension(:),allocatable :: frq

integer ::  nzo,icount
real(kind=wp) :: omega

real(kind=wp) :: t1,t2

integer :: ii,jj,iff,length,length1,length2,itemp

integer, parameter :: nunit=2
complex(kind=wp), parameter :: j=cmplx(0.0_wp,1.0_wp)
complex(kind=wp) :: scl

! the function hostnm appears to be a standard extension.
! it just gets the hostname of the computer - useful when running
! with MPI.
integer :: status,hostnm
character(len=12) :: hostname,hostname0

 ! MPI requirements
integer :: myid, numprocs, ierr, rc, stat(MPI_STATUS_SIZE)
!complex(kind=wp),dimension(:),allocatable :: buffer   ! dimension numprocs

integer :: inode
character(len=20) :: name3     ! weighting for nodes filename
integer :: istart,iend,numfs,numfs0
real (kind=wp) :: xsum
real(kind=wp),dimension(:),allocatable :: nodewt,wts   ! dimension numprocs
integer,dimension(:,:),allocatable :: nodefrqs         ! dimension numprocsX2 

interface

  subroutine ram(zsrc,rg)
    ! ram uses the data stored using the module envdata
    
    use interpolators
    use envdata
    use fld
    use param

    implicit none

    real(kind=wp),dimension(:),intent(in) :: zsrc,rg

  end subroutine ram

end interface
! END OF MANDATORY DECLARATIONS
! READY FOR FIRST EXECUTABLE STATEMENT

 ! Fireup the MPI communication machine
 call MPI_INIT( ierr )
 call MPI_COMM_RANK( MPI_COMM_WORLD, myid, ierr )
 call MPI_COMM_SIZE( MPI_COMM_WORLD, numprocs, ierr )
 !print *, 'Process ', myid, ' of ', numprocs, ' is alive'

status=hostnm(hostname)
!call flush(6)

allocate(nodewt(numprocs))
allocate(rmax(1),zsrc(1))

! myid 0 reads in the data to get started. 
if (myid==0) then

open(nunit,file='in.pe',status='old')

read (2,'(f4.0)')  fc                ! skip the first line - read a dummy
read (2,'(f4.0,1x,f3.1)') fc,Q       ! center frequency and Q
read (2,'(f5.1)') T              ! time window width
read (2,'(f6.1)') zsrc(1)        ! source depth
read (2,'(f12.3)') rmax(1)       ! receiver range
read (2,'(f5.2)') deltaz         ! depth accuracy parameter
read (2,'(f6.2)') deltar         ! range accuracy parameter
read (2,'(i1,1x,i1)') np,nss     ! np -# pade coefficients
                                 ! ns -# stability terms
read (2,'(f7.1)') rs             ! stability range
read (2,'(i2)') dzm              ! dzm - depth decimation
read (2,'(a20)') name1           ! sound speed filename
name1=trim(name1)        ! remove trailing blanks
read (2,'(i1)') iflat            ! 0=no flat earth transform, 1=yes
read (2,'(i1)') ihorz            ! 0=no horizontal linear interpolation, 1=yes
read (2,'(i1)') ibot             ! 0=no bottom, 1=bottom and read a file
read (2,'(a20)') name2           ! bathymetry filename; ignored if ibot=0
name2=trim(name2)        ! remove trailing blanks
read (2,'(i1)') inode            ! 0=equal weights , 1=specify weight from a file
read (2,'(a20)') name3           ! node weighting filename; ignored if inode=0
name3=trim(name3)        ! remove trailing blanks

close(nunit)

if (name1=='munk') then
   print *,'No munk profile for MPI version.' 
   stop
end if

print *
print '(a)','INPUT PARAMETERS:'
print '(a,f10.2)','Center frequency (Hz): ', fc
print '(a,f3.1)','Q: ', Q
print '(a,f5.2)','Bandwidth (f0/Q - Hz): ', fc/Q 
print '(a,f5.1)','Time window width (s): ', T
print '(a,f6.1)','Source depth (m): ', zsrc(1)
print '(a,f12.3)','Range (m): ', rmax(1)
print '(a,f5.2)','Depth accuracy (deltaz, m): ', deltaz
print '(a,f6.2)','Range accuracy (deltar, m): ', deltar
print '(a,i2)','No. of pade coefficients: ', np
print '(a,i2)','No. of stability terms: ', nss
print '(a,f7.1)','Stability range (m): ', rs
print '(a,i2)','Output depth decimation: ', dzm
print '(a,a)','Sound speed filename: ', name1
print '(a,i1)','Flat-earth transform flag: ', iflat
print '(a,i1)','Horizontal interpolation flag: ', ihorz
print '(a,i1)','Ocean bottom flag: ', ibot
print '(a,a)','Ocean bottom filename: ', name2
print *

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! Get the weights for allocating computation for each node.
! Set these weights to equal the speed of computation for each frequency.
! i.e., the smaller the weight, the more frequencies get allocated to that
! node.
if (inode==0) then
   do jj=1,numprocs
        nodewt(jj)=1.0_wp
   enddo
else
   print '(a,i3)','Reading load balancing file of length: ', numprocs
   print *
   open(nunit,file=name3,status='old')
   do jj=1,numprocs
      read(nunit,*) nodewt(jj)
   enddo
   close(nunit)
end if

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! SOUND SPEED READ
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   ! Open and read the sound speed from file name1.
   ! The sound speed file format is the same as eigenray or ray
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
     rewind nunit
     allocate(rp0(nrp0),zw(nzp),cq(nzp,nrp0)) 
     do ii=1,nrp0
        read(nunit,*) dum,rp0(ii)
        rp0(ii)=rp0(ii)*1000.0_wp
        do jj=1,nzp
            read(nunit,*) zw(jj),cq(jj,ii)
        enddo
     end do 
   close(nunit)

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! BATHYMETRY READ
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (ibot==0) then
   ! use default flat bottom
   nb=1
   allocate(rb(nb),zb(nb))
   rb(nb)=0.0_wp
   zb(nb)=maxval(zw)-400.0_wp
    ! Set the ocean ocean floor to be 400 m shallower than deepest sound speed depth.
    ! c.f., attenuation below
else
   ! open and read in the bathymetry from file name2
   print *,'Reading bathymetry file'
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

end if  ! end if for myid=0

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! Now all the basic information is available to do the calculation.
! Broadcast all that information....and let all the nodes calculate
! the other stuff (e.g., sound speed interpolation) needed for the 
! call to ram.  
! The nodes with myid/=0 need to allocate the basic arrays 
! to receive the broadcasts.
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

call MPI_BCAST(rmax,1, MPI_FLOAT,0,MPI_COMM_WORLD,ierr)
call MPI_BCAST(zsrc,1, MPI_FLOAT,0,MPI_COMM_WORLD,ierr)
call MPI_BCAST(deltaz,1,MPI_FLOAT ,0,MPI_COMM_WORLD,ierr)
call MPI_BCAST(deltar,1, MPI_FLOAT,0,MPI_COMM_WORLD,ierr)
call MPI_BCAST(rs,1, MPI_FLOAT,0,MPI_COMM_WORLD,ierr)
call MPI_BCAST(fc,1, MPI_FLOAT,0,MPI_COMM_WORLD,ierr)
call MPI_BCAST(Q,1, MPI_FLOAT,0,MPI_COMM_WORLD,ierr)
call MPI_BCAST(T,1, MPI_FLOAT,0,MPI_COMM_WORLD,ierr)

call MPI_BCAST(np,1, MPI_INTEGER4,0,MPI_COMM_WORLD,ierr)
call MPI_BCAST(nss,1, MPI_INTEGER4,0,MPI_COMM_WORLD,ierr)
call MPI_BCAST(dzm,1, MPI_INTEGER4,0,MPI_COMM_WORLD,ierr)
call MPI_BCAST(iflat,1, MPI_INTEGER4,0,MPI_COMM_WORLD,ierr)
call MPI_BCAST(ihorz,1, MPI_INTEGER4,0,MPI_COMM_WORLD,ierr)

! send the sizes of the arrays first, so that they may be allocated.
call MPI_BCAST(nrp0,1, MPI_INTEGER4,0,MPI_COMM_WORLD,ierr)
call MPI_BCAST(nzp,1, MPI_INTEGER4,0,MPI_COMM_WORLD,ierr)
call MPI_BCAST(nb,1, MPI_INTEGER4,0,MPI_COMM_WORLD,ierr)

! need to have nrp0, nzp, nb to allocate on slave nodes
if (myid/=0) allocate(rp0(nrp0),zw(nzp),cq(nzp,nrp0),rb(nb),zb(nb)) 

      itemp=nrp0
call MPI_BCAST(rp0,itemp, MPI_FLOAT,0,MPI_COMM_WORLD,ierr)
      itemp=nzp
call MPI_BCAST(zw,itemp, MPI_FLOAT,0,MPI_COMM_WORLD,ierr)
      itemp=nrp0*nzp
call MPI_BCAST(cq,itemp, MPI_FLOAT,0,MPI_COMM_WORLD,ierr)

      itemp=nb
call MPI_BCAST(rb,itemp, MPI_FLOAT,0,MPI_COMM_WORLD,ierr)
call MPI_BCAST(zb,itemp, MPI_FLOAT,0,MPI_COMM_WORLD,ierr)

! broadcast node weights
      itemp=numprocs
call MPI_BCAST(nodewt,itemp, MPI_FLOAT,0,MPI_COMM_WORLD,ierr)

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! Sound speed, bathymetry and other information is now transferred to all nodes.
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   ! Now interpolate in range to 10-km intervals (linear), say, if requested,
   ! and interpolate in depth to 1-m intervals (cubic spline),
   
   if (ihorz==1) then
   ! Horizontal
     nrp=nint(rmax(1)/10000.0_wp)
     call linspace(rp, rp0(1),rmax(1),nrp)
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
      if (myid==0) then
        print *
        print *,'Applying flat-earth transformation.'
        print *
      end if
      allocate(eps(nzp))
      eps=zw*invRe
      zw=zw*(1.0_wp+(1.0_wp/2.0_wp)*eps+(1.0_wp/3.0_wp)*eps*eps)
      forall(ii=1:nrp) cw(:,ii)=cw(:,ii)*(1+eps+eps*eps)
      deallocate(eps)
      ! also transform the source depth
      zsrc(1)=zsrc(1)*(1.0_wp+(1.0_wp/2.0_wp)*zsrc(1)*invRe +   &
            (1.0_wp/3.0_wp)*(zsrc(1)*invRe)**2)
   end if

! calculate a mean sound speed
n=size(cw)
c0=sum(cw)/n
ic0=1.0_wp/c0
cmin=minval(cw)     ! minimum sound speed might be better for calculating tdelay

!  DONE WITH SOUND SPEEDS
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! BOTTOM SET UP
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   ! flat earth transformation for bottom
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

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! BOTTOM PROPERTIES MODEL
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

! Sediment speed, density and attenuation are gross approximations.
!  (zw,cw) is depth, water sound speed
!   and (zs,cs) is depth, sediment sound speed

! For the bottom we introduce a simple range-indendent model, but one which
! follows the bathymetry.  The profl subroutine in ram.f90 implements this;
! here the basic model is given.
! Bottom speed, density, and attenuation follow rb,zb

! Sediment layer thickness - meters
sedlayer=300.0_wp

! Sediment sound speed - this will be speed relative to the water sound speed.
! Four values are given:  at the surface, at the bottom, sedlayer-m below the bottom,
! and at the center of the earth...
allocate(cs(4))
cs(1)=0.0_wp   
cs(2)=0.0_wp
cs(3)=200.0_wp 
cs(4)=200.0_wp  ! sound speed increases by 200 m/s sedlayer-m
                                 ! below the sea floor, that is to about 1700 m/s.
! One doesn't want an abrupt change in sound speed, or you get too much reflection.
! The sharper the sound speed gradient, the greater the reflection at the sea floor.

! Sediment density - use only one approximate value:  1.2 seems to be nominal.
allocate(rho(4))
rho(1)=1.2_wp 
rho(2)=1.2_wp
rho(3)=1.2_wp 
rho(4)=1.2_wp

! Sediment attenuation - bottom attenuation is 0.5 dB/wavelength from the ocean 
! surface to the sea floor and increases to 5.0 dB/wavelength sedlayer-m below 
! the ocean floor.   0.5 dB/wavelength is a very large value for low frequency.
allocate(attn(4))
attn(1)=0.5_wp  
attn(2)=0.5_wp
attn(3)=5.0_wp  
attn(4)=5.0_wp

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!                Example using fc=75 Hz, Q=2, and time window T=2.0 
bw=fc/Q        ! 75/2=37.5 Hz bandwidth
fs=4.0_wp*fc   ! 4*75 Hz = 300 Hz sampling frequency
!dt=1.0_wp/fs   ! 1/300 s sampling interval
Nsam=fs*T      ! 300 Hz*2 s = 600 samples
df=fs/Nsam     ! 300/600 = 0.5 Hz frequency interval

! frequency set up
nf1=int((bw-df)/df) + 1  ! (37.5-0.5)/0.5 + 1 = 75 frequencies, one half
nf=2*nf1+1          ! including 0, 151 frequencies altogether.
allocate(frq(nf))
do ii=1,nf1
  frq(ii)=-(nf1-(ii-1))*df + fc
enddo
frq(nf1+1)= fc
do ii=1,nf1
  frq(ii+nf1+1)=ii*df + fc
enddo

do ii=1,(2*nf1+1)
   ! if (myid==0) print *,ii,frq(ii)
   if (frq(ii) <= 0.0) then
       if (myid==0) print *,'Zero or negative frequencies encountered.'
       if (myid==0) print *,'Stopping - fix problem (adjust Q?)'
       call MPI_FINALIZE(rc)
       stop
   endif
enddo

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! Assign each node its frequencies according to its weight
! nf frequecies apportioned according to nodewt(numprocs)
! A simple calculation - we'll have all the nodes do it.

allocate(nodefrqs(numprocs,2), wts(numprocs))

xsum=sum(nodewt)
do ii=1,numprocs
   wts(ii)=xsum/nodewt(ii)
end do
xsum=sum(wts)

iend=0
do ii=1,numprocs
   istart=iend+1
   iend=istart+int(anint(nf*wts(ii)/xsum))-1
   if (ii==numprocs) iend=nf
   nodefrqs(ii,1)=istart
   nodefrqs(ii,2)=iend
enddo

istart=nodefrqs(myid+1,1)
iend=nodefrqs(myid+1,2)
numfs=iend-istart+1

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

! Blocking send and receive to get the print statements in order.
if (myid==0) then
  print *,nf,' total frequencies'
  print '(a)', '  _______________________'
  print '(a)', '  myid  hostname      #fs'
  print '(a)', '  -----------------------'
  print '(a,i3,3x,a,1x,i4)', '  ', myid,hostname,numfs
  do ii=1,(numprocs-1)
    call MPI_RECV(numfs0,1,MPI_INTEGER4,ii,MPI_ANY_TAG,MPI_COMM_WORLD,stat,ierr)
    call MPI_RECV(hostname0,12,MPI_CHARACTER,ii,MPI_ANY_TAG,MPI_COMM_WORLD,stat,ierr)
    print '(a,i3,3x,a,1x,i4)', '  ', ii,hostname0,numfs0
  enddo
  print *,'   '
else
  call MPI_SEND(numfs,1,MPI_INTEGER4,0,icount,MPI_COMM_WORLD,ierr)
  call MPI_SEND(hostname,12,MPI_CHARACTER,0,icount,MPI_COMM_WORLD,ierr)
endif

allocate(psi1(nzo),psif(nzo,nf))
psi1=0.0_wp*psi1
psif=0.0_wp*psif

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! Meat and Potatoes
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
do iff=istart,iend
    call cpu_time(t1)
    frqq=frq(iff)
    ns=nss

    call ram(zsrc,rmax)

! The miracle of fortran95!
    psi1=psi(1:icount:dzm,1)

    omega=2.0_wp*pi*frqq 
    ! 3-D
    scl=exp(j*(omega/c0*rout(1) + pi/4.0_wp))/4.0_wp/pi
    ! 2-D
    ! k0=omega/c0
    !scl=j*exp(j*omega/c0*rout)/sqrt(8.0_wp*pi*k0)
    psif(:,iff)=scl*psi1
!    print '(i5,1x,60(f8.5,1x))',iff,(real(psif(jj,iff)),aimag(psif(jj,iff)),jj=1,30)
    call cpu_time(t2)
    print '(a,i4,1x,i4,a,i3,1x,f8.3,a,f9.2,a)','myid, iff, #fs, frqq = ',  &
         myid,iff-istart+1,' of ',numfs,frqq, '  Time:',nint(100.0_wp*(t2-t1))/100.0_wp,' sec'
end do

allocate(rpsif(nzo,nf),ipsif(nzo,nf))
rpsif=0.0_wp*rpsif
ipsif=0.0_wp*ipsif
do iff=istart,iend
   do jj=1,nzo 
      rpsif(jj,iff)=real(psif(jj,iff))
      ipsif(jj,iff)=aimag(psif(jj,iff))
   end do
end do

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! At this point we're done.
! Now need to save the information for plotting in matlab.
! Need to save:  psif(:,:), rout, c0, zg1(:), fs, wind(:), frq(:), Nsam, nf, Q
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

! Send all the matrices back to node 0 and add 'em up.  
! The size if psif is something like nzo=5000Xnf=500...its not that big, really.

itemp=nzo*nf

! This reduction is far slicker!
allocate(rpsiff(nzo,nf),ipsiff(nzo,nf))
rpsiff=0.0_wp*rpsiff   ! zero out the array for safety!
ipsiff=0.0_wp*ipsiff   ! zero out the array for safety!
call MPI_REDUCE(rpsif,rpsiff,itemp,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD,ierr)
call MPI_REDUCE(ipsif,ipsiff,itemp,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD,ierr)

if (myid==0) then
! write out results

  allocate(zg1(nzo))  ! zg1 is the same for all processors, so only need this for myid=0
  zg1=zg(1:icount:dzm)
!  Remove the flat-earth transform (or most of it, anyways)
  if (iflat==1) then
    allocate(eps(nzo))
    eps=zg1*invRe
    zg1=zg1/(1.0_wp+(1.0_wp/2.0_wp)*eps+(1.0_wp/3.0_wp)*eps*eps)
    deallocate(eps)
  end if

  inquire(iolength=length1) rpsiff(1,:)
  inquire(iolength=length2) zg1(1)
!  print *,'lengths'
!  print *,length1, length2
  length=2*length1+length2   ! We need nf real and nf imaginary values and the depth.
!   length is the record length, an important piece of information
!   Write it out for handy reference:
  open(nunit, form='formatted',file='recl.dat')
  write(nunit,*) length
  close(nunit)

!  Now write out the data to a direct access file:
  open(nunit, access='direct',recl=length,file='psif.dat')

!  Float the integers to real*8, otherwise there will be trouble.
  write(nunit,rec=1) Nsam,real(nf,wp),real(nzo,wp),rout,c0,cmin,fs,Q
  write(nunit,rec=2) frq     ! vector of size nf

  do ii=1,nzo
     write(nunit,rec=ii+2) zg1(ii),((rpsiff(ii,jj)),(ipsiff(ii,jj)),jj=1,nf)
  end do

  close(nunit)

  print *, 'ALL DONE! '
  print *,'recl.dat has the record length of the direct access file.'
  print *,'The direct access file with the parameters and results is psif.dat.'

end if

call MPI_FINALIZE(rc)
stop

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end program peramx

