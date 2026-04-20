module interpolators

implicit none

CONTAINS

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine linspace(z, z1,z2,k)
! z returns k evenly spaced values between z1 and z2, inclusive.

  use kinds

  implicit none

  integer, intent(in) :: k
  real(kind=wp), intent(in) :: z1, z2
  real(kind=wp),dimension(:),allocatable, intent(out) :: z

  integer :: l,n
  real(kind=wp) :: zn

  allocate(z(k))
  l=k-2
  zn=real(k-1,wp)

  forall(n=0:l) z(n+1)=z1+n*(z2-z1)/zn
  z(k)=z2

end subroutine linspace

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function interp1(x,y,xi,val)
! linear interpolation function
! extrapolated values have value val
! only need vector interpolation for ram.f90
!
! Assumes x and xi are both positive and monotonically increasing
! x and xi are not equally spaced
! Which I think is all we run into here.

  use kinds 

  implicit none

  real(kind=wp),dimension(:),allocatable :: interp1
  real(kind=wp),dimension(:) :: x,y,xi

  real(kind=wp),dimension(:),allocatable :: h
  real(kind=wp) :: val,s

  integer :: n,m,ii,jj,m1

  n=size(x); m=size(xi)

  allocate(interp1(m),h(n-1))


  do ii=1,(n-1)
     h(ii)=1.0_wp/(x(ii+1)-x(ii))
  enddo

  m1=1
  do ii=1,m
     ! Find interval index jj such that x(jj) < xi(ii) < x(jj+1)
     ! if xi is outside the range of x's, then fill it with val
     if (xi(ii)<x(1)) then
        interp1(ii)=val
     elseif ((x(1)<=xi(ii)).and.(xi(ii)<=x(n))) then
        do jj=m1,(n-1)
           if ((x(jj)<=xi(ii)).and.(xi(ii)<=x(jj+1))) then
              s = (xi(ii) - x(jj))*h(jj)
              interp1(ii) = y(jj) + s*(y(jj+1)-y(jj))
              m1=jj
              exit
           endif
        enddo
     else
       interp1(ii)=val
     endif
    enddo

end function interp1

end module interpolators
