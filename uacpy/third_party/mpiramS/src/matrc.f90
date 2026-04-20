subroutine matrc

use kinds
use mattri
use param
use profiles

! local variables
integer :: i1,i2,no,nz,id
real(kind=wp),dimension(:),allocatable :: f1,f2

real(kind=wp2) :: c1,c2,c3
complex(kind=wp2) :: d1,d2,d3
complex(kind=wp2),dimension(:),allocatable :: ksq
!,rfact
complex(kind=wp),dimension(:),allocatable :: rfact

! The tridiagonal matrices.
  no=size(pdu)
  nz=size(rhob); nz=nz-2

  ! note that the size of the r's and s's varies, depending on the size of pdu,pdl
  if (.not.allocated(r1)) then
     allocate(r1(nz+2,no),r2(nz+2,no),r3(nz+2,no))
     allocate(s1(nz+2,no),s2(nz+2,no),s3(nz+2,no))
     allocate(f3(nz+2))
  end if
  allocate(f1(nz+2),f2(nz+2))
! zero all the r and s arrays, otherwise they are filled with garbage
! f arrays are overwritten and are o.k.
  r1=cmplx(0.0_wp,0.0_wp,wp); r2=cmplx(0.0_wp,0.0_wp,wp); r3=cmplx(0.0_wp,0.0_wp,wp)
  s1=cmplx(0.0_wp,0.0_wp,wp); s2=cmplx(0.0_wp,0.0_wp,wp); s3=cmplx(0.0_wp,0.0_wp,wp)

! Defined in ram.f, since they dont have to be recalculated each step
!  a1=k0*k0*sixth
!  a2=k0*k0*twothird
!  a3=a1
!  cfact=0.5_wp/deltaz**2
!  dfact=twelfth

  allocate(ksq(nz+2)) 
! New matrices
  forall (id=1:iz) 
     f1(id)=1.0_wp/alpw(id)
     f2(id)=1.0_wp
     f3(id)=alpw(id)
     ksq(id)=ksqw(id)
  end forall

  forall (id=(iz+1):(nz+2)) 
     f1(id)=rhob(id)/alpb(id)
     f2(id)=1.0_wp/rhob(id)
     f3(id)=alpb(id)
     ksq(id)=ksqb(id)
  end forall

! Discretization by Galerkin's method.
  i1=2; i2=nz+1
  do id=i1,i2
     c1=cfact*f1(id)*(f2(id-1)+f2(id))*real(f3(id-1))
     c2=-cfact*f1(id)*(f2(id-1)+2.0_wp2*f2(id)+f2(id+1))*real(f3(id))
     c3=cfact*f1(id)*(f2(id)+f2(id+1))*real(f3(id+1))
     d1=c1+dfact*(ksq(id-1)+ksq(id))
     d2=c2+dfact*(ksq(id-1)+6.0_wp2*ksq(id)+ksq(id+1))
     d3=c3+dfact*(ksq(id)+ksq(id+1))

     r1(id,:)=cmplx(a1,kind=wp)+cmplx(d1,kind=wp)*pdl(:)
     r2(id,:)=cmplx(a2,kind=wp)+cmplx(d2,kind=wp)*pdl(:)
     r3(id,:)=cmplx(a3,kind=wp)+cmplx(d3,kind=wp)*pdl(:)
     s1(id,:)=cmplx(a1,kind=wp)+cmplx(d1,kind=wp)*pdu(:)
     s2(id,:)=cmplx(a2,kind=wp)+cmplx(d2,kind=wp)*pdu(:)
     s3(id,:)=cmplx(a3,kind=wp)+cmplx(d3,kind=wp)*pdu(:)
  end do

  deallocate(f1,f2,ksq)
 
! The matrix decomposition.
  allocate(rfact(no)) ; rfact=cmplx(0.0_wp,0.0_wp,wp) 
  do id=i1,iz
    rfact=cmplx(1.0_wp2/(r2(id,:)-r1(id,:)*r3(id-1,:)),kind=wp)
    r1(id,:)=r1(id,:)*rfact
    r3(id,:)=r3(id,:)*rfact
    s1(id,:)=s1(id,:)*rfact
    s2(id,:)=s2(id,:)*rfact
    s3(id,:)=s3(id,:)*rfact
  end do
! According to gprof, this loop above might amount to 5% of the computation time.
 
  do id=i2,(iz+2),-1
    rfact=cmplx(1.0_wp2/(r2(id,:)-r3(id,:)*r1(id+1,:)),kind=wp)
    r1(id,:)=r1(id,:)*rfact
    r3(id,:)=r3(id,:)*rfact
    s1(id,:)=s1(id,:)*rfact
    s2(id,:)=s2(id,:)*rfact
    s3(id,:)=s3(id,:)*rfact
  end do

  r2(iz+1,:)=r2(iz+1,:)-r1(iz+1,:)*r3(iz,:)
  r2(iz+1,:)=r2(iz+1,:)-r3(iz+1,:)*r1(iz+2,:)
  r2(iz+1,:)=1.0_wp/r2(iz+1,:)

  deallocate(rfact)

end subroutine matrc
