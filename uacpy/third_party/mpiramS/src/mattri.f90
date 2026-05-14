module mattri
! The vectors and matrices used between matrc and solvetri

use kinds

real(kind=wp2), parameter :: dfact=0.0833333333333333  !  dfact=twelfth
real(kind=wp2) :: cfact, a1, a2, a3

complex(kind=wp),dimension(:),allocatable :: uu,f3
complex(kind=wp),dimension(:,:),allocatable :: r1, r2, r3, s1, s2, s3

!$OMP THREADPRIVATE (cfact,a1,a2,a3,uu,f3,r1,r2,r3,s1,s2,s3)

end module mattri
