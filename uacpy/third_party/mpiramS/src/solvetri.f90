subroutine solvetri

use kinds
use mattri
use param    ! carries the all-important iz

implicit none

! local variables
integer :: nz,no,ind,nz1,ii,jj
! real :: eps
complex(kind=wp),dimension(:),allocatable ::  v

!  eps=1.0e-30_wp

  nz=size(r1,1)
  no=size(r1,2)    ! no is just np, the number of pade coefficients, i.e., 4. 

  allocate(v(nz)); v=cmplx(0.0_wp,0.0_wp,wp)

  nz=nz-2
  nz1=nz+1

  do jj=1,no

    ! The right side.
!    forall (ind=2:nz1)  v(ind)= s1(ind,jj)*uu(ind-1) + s2(ind,jj)*uu(ind) + s3(ind,jj)*uu(ind+1)  
    do ind=2,nz1
       v(ind)= s1(ind,jj)*uu(ind-1) + s2(ind,jj)*uu(ind) + s3(ind,jj)*uu(ind+1) 
    end do
! According to gprof 41% of computation time is in the loop above
    
    ! The elimination steps.
    do ii=3,iz
      v(ii)=v(ii)-r1(ii,jj)*v(ii-1) 
    end do
! According to gprof 16% of computation time is in the loop above
    do ii=nz,iz+2,-1
      v(ii)=v(ii)-r3(ii,jj)*v(ii+1) 
    end do
! According to gprof 3% of computation time is in the loop above

    uu(iz+1)=(v(iz+1)-r1(iz+1,jj)*v(iz)-r3(iz+1,jj)*v(iz+2))*r2(iz+1,jj) 

    ! The back substitution steps.
    do ii=iz,2,-1
      uu(ii)=v(ii)-r3(ii,jj)*uu(ii+1) 
    end do
! According to gprof 16% of computation time is in the loop above
    do ii=iz+2,nz+1
      uu(ii)=v(ii)-r1(ii,jj)*uu(ii-1) 
    end do
! According to gprof 3% of computation time is in the loop above
  end do

  deallocate(v)

! uu is overwritten and returned with the answer

end subroutine solvetri

