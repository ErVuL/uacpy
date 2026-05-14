module fld
! The vectors and matrices used between peramx and ram

use kinds

real(kind=wp) :: frqq
real(kind=wp),dimension(:),allocatable :: rout
complex(kind=wp),dimension(:,:),allocatable :: psi

!$OMP THREADPRIVATE (frqq,rout,psi)

end module fld
