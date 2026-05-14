module param
! A collection of parameters.
! Modules such as these take the place of "common" blocks, but
! are meant to be better - less error prone and easier to manage.

use kinds

implicit none

real(kind=wp),parameter :: pi=3.141592653589793_wp, Re=6378137.0_wp, invRe=1.0_wp/Re

integer :: np,ns,ip     ! # pade coefficients, # stability terms, and
                        ! self-starter or split-step flag

integer :: iz         ! iz is the index for the sea floor depth

real(kind=wp) :: dre,k0    ! epade range increment, wavenumber

real(kind=wp) :: zmax,c0,ic0,deltaz,deltar,rs
! max depth, mean sound speed, mean slowness, depth increment, 
! range increment, stability range

!$OMP THREADPRIVATE (ns,ip,iz,dre,k0)

end module param

