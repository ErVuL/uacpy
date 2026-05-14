module envdata
! sound speed and bathymetry data and parameters
! Modules such as these take the place of "common" blocks, but
! are meant to be better - less error prone.

use kinds

implicit none

real(kind=wp) :: sedlayer
integer :: nzs                                             ! number of sediment depth points
real(kind=wp),dimension(:),allocatable :: zg               ! depth grid with deltaz spacing
real(kind=wp),dimension(:),allocatable :: rb,zb            ! bathymetry
real(kind=wp),dimension(:),allocatable :: rp,zw            ! range and depths of sound speeds
real(kind=wp),dimension(:,:),allocatable ::  cw            ! sound speeds, etc.

! Bottom sediment properties.
! When isedrd==0 (range-independent): cs(nzs,1), rho(nzs,1), attn(nzs,1) — single profile.
! When isedrd==1 (range-dependent):   cs(nzs,nrp_sed), rho(nzs,nrp_sed), attn(nzs,nrp_sed)
!   with rp_sed(nrp_sed) giving the range points (in metres).
integer :: isedrd                                          ! 0=range-indep, 1=range-dep sediment
integer :: nrp_sed                                         ! number of sediment range profiles
real(kind=wp),dimension(:),   allocatable :: rp_sed        ! sediment range points (m)
real(kind=wp),dimension(:,:), allocatable :: cs,rho,attn   ! bottom properties (nzs, nrp_sed)

end module envdata


