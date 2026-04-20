subroutine epade
!subroutine epade(pdu,pdl, np,ns,ip,k0,dr)
! function [pdu, pdl]=epade( np, ns, ip, k0, dr)
!
! np number of pade coefficients
! ns stability constraints
! ip self starter=0, split-step=1
! k0 (rad/m)
! dr range step (m)

! It develops that most everything here should be 
! in double precision.

use kinds
use profiles
use param

implicit none

integer :: ii,jj,n
real(kind=wp) ::  dr
real(kind=wp2) :: z1, sig, nu, alp

real(kind=wp2), dimension(:), allocatable  :: fact
complex(kind=wp2), dimension(:), allocatable  :: b,dg,dh1,dh2,dh3
real(kind=wp2), dimension(:,:), allocatable  :: bin
complex(kind=wp2), dimension(:,:), allocatable  :: a

if (allocated(pdu)) deallocate(pdu,pdl)
allocate(pdu(np),pdl(np))

  dr=dre
  sig=dble(k0*dr)
  n=2*np

  if (ip==1) then   ! Split-step Pade approximation
    nu=0.0_wp2
    alp=0.0_wp2
  elseif (ip==2) then   ! Self-starter approximation (2-D)
    nu=1.0_wp2
    alp=-0.5_wp2
  else       ! Self-starter approximation (3-D)
    nu=1.0_wp2
    alp=-0.25_wp2
  end if

allocate(fact(n))
! The factorials.
fact(1)=1.0_wp2
  do ii=2,n
    fact(ii)=real(ii,wp2)*fact(ii-1)
  end do

allocate(bin(n+1,n+1))
bin=0.0_wp2*bin
! The binomial coefficients.
  do ii=1,(n+1)
    bin(ii,1)=1.0_wp2
    bin(ii,ii)=1.0_wp2
  end do
  do ii=3,(n+1)
    do jj=2,(ii-1)
      bin(ii,jj)=bin(ii-1,jj-1)+bin(ii-1,jj)
    end do
  end do

! The accuracy constraints.
  call deriv(dg,dh1,dh2,dh3, n,sig,alp,nu,bin)

  allocate(a(n,n))
  allocate(b(n))
  b(1:n)=dg(2:(n+1))
 
  ! zero a!
  a=0.0_wp2*a

  do ii=1,n
    if ((2*ii-1)<=n) a(ii,2*ii-1)=fact(ii)
    do jj=1,ii
      if ((2*jj)<=n) a(ii,2*jj)=-bin(ii+1,jj+1)*fact(jj)*dg(ii-jj+1)
    end do
  end do

! The stability constraints.
  if (ns>=1) then
    z1=-3.0_wp2
    b(n)=-1.0_wp2
    forall(ii=1:np) a(n,2*ii-1)=z1**ii
    forall(ii=1:np) a(n,2*ii)=0.0_wp2
  end  if

  if (ns>=2) then
    z1=-1.5_wp2
    b(n-1)=-1.0_wp2
    forall(ii=1:np) a(n-1,2*ii-1)=z1**ii
    forall(ii=1:np) a(n-1,2*ii)=0.0_wp2
  end if

  call gauss(a,b)   ! takes the place of b=a\b; uses gaussian elimination routine.
  
  dh1(1)=1.0_wp2
  forall(ii=1:np) dh1(ii+1)=b(2*ii-1)
  call fndrt(dh2, dh1,np)
  forall(ii=1:np) pdu(ii)=cmplx(-1.0_wp2/dh2(ii),kind=wp)

  dh1(1)=1.0_wp2
  forall(ii=1:np) dh1(ii+1)=b(2*ii)
  call fndrt(dh2, dh1,np)
  forall(ii=1:np) pdl(ii)=cmplx(-1.0_wp2/dh2(ii),kind=wp)

  deallocate(bin,fact,dg,dh1,dh2,dh3,a,b)

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CONTAINS
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine deriv(dg,dh1,dh2,dh3, n,sig,alp,nu,bin)
! The derivatives of the operator function at x=0.

use kinds

implicit none

integer, intent(in) :: n
real(kind=wp2), intent(in) :: sig,alp,nu
real(kind=wp2), dimension(:,:), allocatable, intent(in) :: bin
complex(kind=wp2), dimension(:), allocatable, intent(out)  :: dg,dh1,dh2,dh3
complex(kind=wp2) :: ci

integer :: ii,jj
real(kind=wp2) :: exp1, exp2, exp3

  allocate(dg(n+1),dh1(n),dh2(n),dh3(n))

  ci=cmplx(0.0_wp2,1.0_wp2,wp2)

  dh1(1)=0.5_wp2*ci*sig
  dh2(1)=alp
  dh3(1)=-2.0_wp2*nu
  exp1=-0.5_wp2
  exp2=-1.0_wp2
  exp3=-1.0_wp2

  do ii=2,n
    dh1(ii)=dh1(ii-1)*exp1
    dh2(ii)=dh2(ii-1)*exp2
    dh3(ii)=-nu*dh3(ii-1)*exp3
    exp1=exp1-1.0_wp2
    exp2=exp2-1.0_wp2
    exp3=exp3-1.0_wp2
  end do

  dg(1)=1.0_wp2
  dg(2)=dh1(1)+dh2(1)+dh3(1)
  do ii=2,n
    dg(ii+1)=dh1(ii)+dh2(ii)+dh3(ii)
    do jj=1,ii-1
      dg(ii+1)=dg(ii+1)+bin(ii,jj)*(dh1(jj)+dh2(jj)+dh3(jj))*dg(ii-jj+1)
    end do
  end do

end subroutine deriv

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine fndrt(z, a,n)
! The root-finding subroutine. 

use kinds

implicit none

integer, intent(in) :: n
complex(kind=wp2), dimension(:), allocatable, intent(inout) :: a
complex(kind=wp2), dimension(:), allocatable, intent(out)  :: z

integer :: ii,k

real(kind=wp2) :: err
complex(kind=wp2) :: root, root1

allocate(z(n))

  if (n==1) then
    z(1)=-a(1)/a(2)
    return
  end if

  if (n/=2) then
    do k=n,3,-1
      ! Obtain an approximate root.
      root=0.0_wp2
      err=1.0e-12_wp2
      call guerre(root1, a,k,root,err,1000)
      root=root1
  
      ! Refine the root by iterating five more times.
      err=0.0_wp2
      call guerre(root1,a,k,root,err,5)
      root=root1
      z(k)=root
  
      ! Divide out the factor (z-root).
      do ii=k,1,-1
        a(ii)=a(ii)+root*a(ii+1)
      end do
      do ii=1,k
        a(ii)=a(ii+1)
      end do
    end do
  end if

! Solve the quadratic equation.
  z(2)=0.5_wp2*(-a(2)+sqrt(a(2)**2-4.0_wp2*a(1)*a(3)))/a(3)
  z(1)=0.5_wp2*(-a(2)-sqrt(a(2)**2-4.0_wp2*a(1)*a(3)))/a(3)

end subroutine fndrt

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine guerre(z, a,n,guess,err,nter)
!     This subroutine finds a root of a polynomial of degree n > 2 by 
!     Laguerre's method.

use kinds

implicit none

integer, intent(in) :: n,nter
real(kind=wp2), intent(in) :: err
complex(kind=wp2), intent(in) :: guess
complex(kind=wp2), dimension(:), allocatable, intent(in) :: a
complex(kind=wp2), intent(out)  :: z

integer :: ii,iter,jter
real(kind=wp2) :: epsb,amp1,amp2,rn
complex(kind=wp2) :: dz,f,g,h,p,pz,pzz,ci
complex(kind=wp2), dimension(:), allocatable :: az, azz

  ci=cmplx(0.0_wp2,1.0_wp2,kind=wp2)
  epsb=1.0e-20_wp2
  z=guess
  rn=real(n,wp2)

! The coefficients of p'(z) and p''(z).
  allocate(az(n),azz(n-1))

  forall(ii=1:n) az(ii)=real(ii,wp2)*a(ii+1)
  forall(ii=1:n-1) azz(ii)=real(ii,wp2)*az(ii+1)

  iter=0; jter=1
  do 
    p=a(n)+a(n+1)*z

    do ii=(n-1),1,-1
      p=a(ii)+z*p
    end do
    if (abs(p)<epsb) exit

    pz=az(n-1)+az(n)*z
    do ii=(n-2),1,-1
      pz=az(ii)+z*pz
    end do

    pzz=azz(n-2)+azz(n-1)*z
    do ii=(n-3),1,-1
      pzz=azz(ii)+z*pzz
    end do

    ! The Laguerre perturbation.
    f=pz/p
    g=f**2-pzz/p
    h=sqrt((rn-1.0_wp2)*(rn*g-f**2))
    amp1=abs(f+h)
    amp2=abs(f-h)
    if (amp1>amp2) then
      dz=-rn/(f+h)
    else
      dz=-rn/(f-h)
    end if

    iter=iter+1

    ! Rotate by 90 degrees to avoid limit cycles. 
    jter=jter+1
    if (jter==10) then
      jter=1
      dz=dz*ci
    end if
    z=z+dz

    if ((abs(dz)>err).and.(iter<nter)) then
      continue
    else
      exit
    end if
  end do

  deallocate(az,azz)

end subroutine guerre

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

!     Gaussian elimination.
!
subroutine gauss(a,b)

      use kinds

      implicit none

      integer :: i, j, n, k
      complex(kind=wp2), dimension(:), allocatable, intent(inout) :: b
      complex(kind=wp2), dimension(:,:), allocatable, intent(inout) :: a

      n=size(b)
!
!     Downward elimination.
!
      do 4 i=1,n
        if(i.lt.n)call pivot(n,i,a,b)
        a(i,i)=1.0_wp2/a(i,i)
        b(i)=b(i)*a(i,i)
        if(i.lt.n)then
          do 1 j=i+1,n
            a(i,j)=a(i,j)*a(i,i)
    1     end do
          do 3 k=i+1,n
            b(k)=b(k)-a(k,i)*b(i)
            do 2 j=i+1,n
              a(k,j)=a(k,j)-a(k,i)*a(i,j)
    2       end do
    3     end do
        end if
    4 end do
!
!     Back substitution.
!
      do 6 i=n-1,1,-1
        do 5 j=i+1,n
          b(i)=b(i)-a(i,j)*b(j)
    5   end do
    6 end do
!
      return
end subroutine gauss
!
!     Rows are interchanged for stability.
!
subroutine pivot(n,i,a,b)

      use kinds

      implicit none

      integer, intent(in) :: i,n
      complex(kind=wp2), dimension(:), allocatable, intent(inout) :: b
      complex(kind=wp2), dimension(:,:), allocatable, intent(inout) :: a

      integer :: j, i0
      real(kind=wp2) :: amp0, amp
      complex(kind=wp2) :: temp
!
      i0=i
      amp0=cdabs(a(i,i))
      do 1 j=i+1,n
        amp=cdabs(a(j,i))
        if(amp.gt.amp0)then
          i0=j
          amp0=amp
        end if
    1 end do

      if(i0.eq.i)return
!
      temp=b(i)
      b(i)=b(i0)
      b(i0)=temp

      do 2 j=i,n
        temp=a(i,j)
        a(i,j)=a(i0,j)
        a(i0,j)=temp
    2 end do
!
      return
end subroutine pivot

end subroutine epade
