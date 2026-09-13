module profiles

   use kinds

   real(kind=wp),dimension(:),allocatable :: cwg, rhob, alpw, alpb
   complex(kind=wp),dimension(:),allocatable :: ksqb, pdu, pdl, ksqw

!$OMP THREADPRIVATE (pdu,pdl,cwg,rhob,alpw,alpb,ksqw,ksqb)

end module profiles
