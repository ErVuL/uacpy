# Modifications to Vendored Native Codebases

This document summarizes all changes applied to the original Fortran/C source
code shipped with uacpy, with exact diffs.

---

## Acoustics Toolbox (Bellhop, Kraken, Scooter, Bounce, SPARC)

### `KrakenField/field.f90` -- out-of-bounds sentinel fix

`EvaluateADMod` and `EvaluateCMMod` both declare their `rProf` argument as
`rProf(NProf + 1)` and use the extra element as a sentinel value.  However,
`ReadVector` (in `misc/SourceReceiverPositions.f90`) only allocates
`MAX(3, NProf)` elements.  When `NProf >= 3` the access to `rProf(NProf + 1)`
reads past the end of the array.

**Fix:** after `ReadVector` returns, reallocate `rProf` with `NProf + 1`
elements and set `rProf(NProf + 1) = HUGE(rProf(1))`.

```diff
@@ -106,6 +106,16 @@
   CALL ReadVector( NProf, RProf, 'Profile ranges, RProf', 'km' )
   RProf = RProf / 1000.0   ! convert m back to km (undoing what ReadVector did)
 
+  ! EvaluateAD/EvaluateCM access rProf( NProf + 1 ) as a sentinel.
+  ! ReadVector only allocates MAX(3, NProf) elements, so extend by one.
+  BLOCK
+    REAL (KIND=8), ALLOCATABLE :: rProfTmp( : )
+    ALLOCATE( rProfTmp( NProf + 1 ) )
+    rProfTmp( 1 : NProf ) = rProf( 1 : NProf )
+    rProfTmp( NProf + 1 ) = HUGE( rProf( 1 ) )
+    CALL MOVE_ALLOC( rProfTmp, rProf )
+  END BLOCK
+
   IF ( NProf == 1      ) THEN
      WRITE( PRTFile, * ) 'Range-independent calculation'
   ELSE
```

---

## bellhopcuda (C++/CUDA port of Bellhop)

### `config/cuda/SetupCUDA.cmake` -- CUDA arch detection for modern/laptop GPUs

bellhopcuda picks the CUDA compute capability by looking up the GPU name
(from `nvidia-smi -L`) in a hardcoded table.  Two issues made the default
configure fail on current hardware:

1. The table is missing most laptop variants (e.g. `RTX 3060 Laptop GPU`,
   `RTX 4070 Laptop GPU`) and Ada-Lovelace desktop cards (`RTX 4060`,
   `RTX 4070`, `RTX 4070 Ti`).  When the name is not found the configure
   errors out with `CUDA compute capability of GPU <name> unknown!`.
2. The validation in `get_gencode_args` rejected any `CUDA_ARCH_OVERRIDE`
   greater than 75 -- even though the table itself already contained
   entries for compute 86 and 89.  This made the documented
   `-DCUDA_ARCH_OVERRIDE=86` escape hatch unusable.

**Fix:** widen the override validation range (75 -> 120 to cover Ampere,
Ada-Lovelace, Hopper, and Blackwell) and extend the GPU table with the
laptop + modern desktop variants as a safety net.  `install.sh` /
`install.bat` now auto-detect the compute capability via
`nvidia-smi --query-gpu=compute_cap` and pass it as
`-DCUDA_ARCH_OVERRIDE=<XX>`, so the name table is only a fallback.

```diff
@@ -61 +61 @@
-	set(GPU_DATABASE "30:GTX 770,GTX 760,...;86:RTX A5500 Laptop GPU,RTX 3090 Ti,RTX 3090,RTX 3080 Ti,RTX 3080,RTX 3070 Ti,RTX 3070,RTX 3060 Ti,RTX 3060,RTX 3050 Ti,RTX 3050;89:RTX A6000,RTX 4090,RTX 4080,RTX 4070 Ti,RTX 4060 Laptop GPU")
+	set(GPU_DATABASE "30:GTX 770,GTX 760,...;86:RTX A5500 Laptop GPU,RTX 3090 Ti,RTX 3090,RTX 3080 Ti,RTX 3080 Ti Laptop GPU,RTX 3080,RTX 3080 Laptop GPU,RTX 3070 Ti,RTX 3070 Ti Laptop GPU,RTX 3070,RTX 3070 Laptop GPU,RTX 3060 Ti,RTX 3060,RTX 3060 Laptop GPU,RTX 3050 Ti,RTX 3050 Ti Laptop GPU,RTX 3050,RTX 3050 Laptop GPU;89:RTX A6000,RTX 4090,RTX 4090 Laptop GPU,RTX 4080,RTX 4080 Laptop GPU,RTX 4070 Ti,RTX 4070 Ti Laptop GPU,RTX 4070,RTX 4070 Laptop GPU,RTX 4060 Ti,RTX 4060,RTX 4060 Laptop GPU,RTX 4050 Laptop GPU")
@@ -80 +80 @@
-            if(ARCH MATCHES "^[0-9]+$" AND ARCH GREATER_EQUAL 30 AND ARCH LESS_EQUAL 75)
+            if(ARCH MATCHES "^[0-9]+$" AND ARCH GREATER_EQUAL 30 AND ARCH LESS_EQUAL 120)
```

### `src/mode/tl.cpp` -- SHDFIL field widths match Fortran spec

`WriteHeader` / `ReadOutTL` in `src/mode/tl.cpp` write several fields of
the `.shd` binary at the wrong width compared to the Fortran
Acoustics-Toolbox format bellhopcuda claims to be a drop-in replacement
for.  The Fortran layout (declared in
`Acoustics-Toolbox/misc/SourceReceiverPositions.f90` and
`misc/RWSHDFile.f90`) is:

| field              | Fortran type     | bytes |
| ------------------ | ---------------- | ----- |
| `Sx, Sy`           | `REAL(KIND=8)`   | 8     |
| `Sz, Rz`           | `REAL(KIND=4)`   | 4     |
| `Rr, theta`        | `REAL(KIND=8)`   | 8     |
| `freqVec, freq0`   | `REAL(KIND=8)`   | 8     |
| `atten`            | `REAL(KIND=8)`   | 8     |

Upstream bellhopcuda stores `Pos->Sx, Sy, Rr, theta` as `float*` and
down-casts `freq0, atten` to `float` on write, emitting 4-byte values
for all of them.  Tools that follow the Fortran spec (uacpy's
`read_shd_bin` / `read_shd_file`, the Matlab `read_shd_bin.m` shipped
with the Acoustics Toolbox) then read garbage bytes from record 10
(receiver ranges) and fail to plot the field with pcolormesh
monotonicity violations (e.g. `ranges[:3] ≈ [1.5e+15, 1.1e+17, …]`).

**Fix:** convert these arrays to `double` on write (and read them back
as `double`, down-casting to the existing `float*` storage on read).
`Sz, Rz` stay `float` because the Fortran spec also declares them
`REAL(KIND=4)`.  `LRecl` is adjusted for the new 8-byte widths.

```diff
@@ -36,22 +36,25 @@
 template<bool O3D> inline void WriteHeader(
-    const bhcParams<O3D> &params, DirectOFile &SHDFile, float atten,
+    const bhcParams<O3D> &params, DirectOFile &SHDFile, double atten,
     const std::string &PlotType)
 {
+    // SHDFIL format: freq0, atten, Sx, Sy, Rr, theta, freqVec are REAL*8 (double);
+    // Sz, Rz are REAL*4 (float). Matches Acoustics-Toolbox/misc/SourceReceiverPositions.f90.
     const Position *Pos      = params.Pos;
     ...
     int32_t LRecl = 84; // 4 for LRecl, 80 for Title
-    LRecl = bhc::max(LRecl, 2 * freqinfo->Nfreq * (int32_t)sizeof(freqinfo->freqVec[0]));
-    LRecl = bhc::max(LRecl, Pos->Ntheta * (int32_t)sizeof(Pos->theta[0]));
+    LRecl = bhc::max(LRecl, freqinfo->Nfreq * (int32_t)sizeof(double));
+    LRecl = bhc::max(LRecl, Pos->Ntheta * (int32_t)sizeof(double));
     if(!isTL) {
-        LRecl = bhc::max(LRecl, Pos->NSx * (int32_t)sizeof(Pos->Sx[0]));
-        LRecl = bhc::max(LRecl, Pos->NSy * (int32_t)sizeof(Pos->Sy[0]));
+        LRecl = bhc::max(LRecl, Pos->NSx * (int32_t)sizeof(double));
+        LRecl = bhc::max(LRecl, Pos->NSy * (int32_t)sizeof(double));
     }
     LRecl = bhc::max(LRecl, Pos->NSz * (int32_t)sizeof(Pos->Sz[0]));
     LRecl = bhc::max(LRecl, Pos->NRz * (int32_t)sizeof(Pos->Rz[0]));
+    LRecl = bhc::max(LRecl, Pos->NRr * (int32_t)sizeof(double));
     LRecl = bhc::max(LRecl, Pos->NRr * (int32_t)sizeof(cpxf));
     ...
-    DOFWRITEV(SHDFile, (float)freqinfo->freq0);
-    DOFWRITEV(SHDFile, atten);
+    DOFWRITEV(SHDFile, (double)freqinfo->freq0);
+    DOFWRITEV(SHDFile, (double)atten);
     SHDFile.rec(3);
-    DOFWRITE(SHDFile, freqinfo->freqVec, freqinfo->Nfreq * sizeof(freqinfo->freqVec[0]));
+    for(int32_t i = 0; i < freqinfo->Nfreq; ++i)
+        DOFWRITEV(SHDFile, (double)freqinfo->freqVec[i]);
     SHDFile.rec(4);
-    DOFWRITE(SHDFile, Pos->theta, Pos->Ntheta * sizeof(Pos->theta[0]));
+    for(int32_t i = 0; i < Pos->Ntheta; ++i)
+        DOFWRITEV(SHDFile, (double)Pos->theta[i]);

     if(!isTL) {
         SHDFile.rec(5);
-        DOFWRITE(SHDFile, Pos->Sx, Pos->NSx * sizeof(Pos->Sx[0]));
+        for(int32_t i = 0; i < Pos->NSx; ++i)
+            DOFWRITEV(SHDFile, (double)Pos->Sx[i]);
         SHDFile.rec(6);
-        DOFWRITE(SHDFile, Pos->Sy, Pos->NSy * sizeof(Pos->Sy[0]));
+        for(int32_t i = 0; i < Pos->NSy; ++i)
+            DOFWRITEV(SHDFile, (double)Pos->Sy[i]);
     } else {
         SHDFile.rec(5);
-        DOFWRITEV(SHDFile, Pos->Sx[0]);
-        DOFWRITEV(SHDFile, Pos->Sx[Pos->NSx - 1]);
+        DOFWRITEV(SHDFile, (double)Pos->Sx[0]);
+        DOFWRITEV(SHDFile, (double)Pos->Sx[Pos->NSx - 1]);
         SHDFile.rec(6);
-        DOFWRITEV(SHDFile, Pos->Sy[0]);
-        DOFWRITEV(SHDFile, Pos->Sy[Pos->NSy - 1]);
+        DOFWRITEV(SHDFile, (double)Pos->Sy[0]);
+        DOFWRITEV(SHDFile, (double)Pos->Sy[Pos->NSy - 1]);
     }
     ...
     SHDFile.rec(9);
-    DOFWRITE(SHDFile, Pos->Rr, Pos->NRr * sizeof(Pos->Rr[0]));
+    for(int32_t i = 0; i < Pos->NRr; ++i)
+        DOFWRITEV(SHDFile, (double)Pos->Rr[i]);
 }
```

`WriteOutTL`'s local `atten` must match the new parameter type:

```diff
-    real atten = FL(0.0);
+    double atten = 0.0;
```

And `ReadOutTL` mirrors the widths -- read as `double`, down-cast to
the existing `float*` storage on store:

```diff
-    float temp;
+    double temp;
     DIFREADV(SHDFile, temp);
-    freqinfo->freq0 = temp;
-    float atten;
+    freqinfo->freq0 = (real)temp;
+    double atten;
     DIFREADV(SHDFile, atten);
     ...
-    DIFREAD(SHDFile, freqinfo->freqVec, freqinfo->Nfreq * sizeof(freqinfo->freqVec[0]));
+    for(int32_t i = 0; i < freqinfo->Nfreq; ++i) {
+        double v; DIFREADV(SHDFile, v); freqinfo->freqVec[i] = (real)v;
+    }
     ...
-    DIFREAD(SHDFile, Pos->theta, Pos->Ntheta * sizeof(Pos->theta[0]));
+    for(int32_t i = 0; i < Pos->Ntheta; ++i) {
+        double v; DIFREADV(SHDFile, v); Pos->theta[i] = (float)v;
+    }
     ...
-    DIFREAD(SHDFile, Pos->Sx, Pos->NSx * sizeof(Pos->Sx[0]));
+    for(int32_t i = 0; i < Pos->NSx; ++i) {
+        double v; DIFREADV(SHDFile, v); Pos->Sx[i] = (float)v;
+    }
     ...
-    DIFREAD(SHDFile, Pos->Rr, Pos->NRr * sizeof(Pos->Rr[0]));
+    for(int32_t i = 0; i < Pos->NRr; ++i) {
+        double v; DIFREADV(SHDFile, v); Pos->Rr[i] = (float)v;
+    }
```

(Same pattern for `Sy` in both rectilinear and `TL` branches, omitted
above for brevity.)

---

## mpiramS (RAM parabolic-equation model)

### `Makefile` -- portability

Removes the hardcoded compiler path and Intel micro-architecture target so the
build works on any machine with gfortran installed.

```diff
@@ -8,7 +8,7 @@
 
 ###########################################
 # Gnu g77/gfortran options (64 bit)
-FC = /usr/bin/gfortran-13
+FC = gfortran
 
 #FFLAGS = -march=native -mtune=native -fopenmp -m64 -mfpmath=sse -I $(MODDIR) -Wall -finline-functions -ffast-math -fno-strength-reduce -falign-functions=2  -O3 -fomit-frame-pointer 
 #FFLAGS = -g -pg -march=native -fopenmp -m64 -mfpmath=sse -I $(MODDIR) -Wall 
@@ -19,8 +19,8 @@
 #LDFLAGS = -fopenmp -march=native -mtune=native
 #LDFLAGS = -g -pg -fopenmp -march=native -mtune=native 
 
-FFLAGS = -Ofast -march=icelake-client -fopenmp -I $(MODDIR) -Wall -fuse-linker-plugin
-LDFLAGS = -Ofast -fopenmp -march=icelake-client -flto
+FFLAGS = -Ofast -march=native -fopenmp -I $(MODDIR) -Wall -fuse-linker-plugin
+LDFLAGS = -Ofast -fopenmp -march=native -flto
```

### `src/kinds.f90` -- single to double precision

Every `real(kind=wp)` variable in mpiramS inherits from this parameter.  The PE
algorithm accumulates phase over thousands of range steps; at low frequencies
and long ranges single-precision arithmetic loses significance in the complex
exponential `exp(i k dr)`.  The original code already used double precision
(`wp2`) for selected critical operations, acknowledging the limitation.  The
Python reader (`io/mpirams_reader.py`) expects float64 output.

```diff
@@ -1,6 +1,6 @@
 module kinds
 
-integer,parameter :: wp = kind(1.0e0)
+integer,parameter :: wp = kind(1.0d0)
  integer,parameter :: wp2 = kind(1.0d0)
 
  end module kinds
```

### `src/matrc.f90` -- safe complex-zero initialization

Fortran `allocate` does not initialise memory.  Multiplying by zero does not
produce zero when the operand is NaN or Inf (`0 * NaN = NaN`).

```diff
@@ -28,8 +28,8 @@
   allocate(f1(nz+2),f2(nz+2))
 ! zero all the r and s arrays, otherwise they are filled with garbage
 ! f arrays are overwritten and are o.k.
-  r1=0.0_wp*r1; r2=0.0_wp*r2; r3=0.0_wp*r3
-  s1=0.0_wp*s1; s2=0.0_wp*s2; s3=0.0_wp*s3
+  r1=cmplx(0.0_wp,0.0_wp,wp); r2=cmplx(0.0_wp,0.0_wp,wp); r3=cmplx(0.0_wp,0.0_wp,wp)
+  s1=cmplx(0.0_wp,0.0_wp,wp); s2=cmplx(0.0_wp,0.0_wp,wp); s3=cmplx(0.0_wp,0.0_wp,wp)
 
 ! Defined in ram.f, since they dont have to be recalculated each step
 !  a1=k0*k0*sixth
@@ -75,7 +75,7 @@
   deallocate(f1,f2,ksq)
  
 ! The matrix decomposition.
-  allocate(rfact(no)) ; rfact=0.0_wp*rfact 
+  allocate(rfact(no)) ; rfact=cmplx(0.0_wp,0.0_wp,wp)
   do id=i1,iz
     rfact=cmplx(1.0_wp2/(r2(id,:)-r1(id,:)*r3(id-1,:)),kind=wp)
     r1(id,:)=r1(id,:)*rfact
```

### `src/solvetri.f90` -- safe complex-zero initialization

Same `0 * NaN` fix as `matrc.f90`.

```diff
@@ -16,7 +16,7 @@
   nz=size(r1,1)
   no=size(r1,2)    ! no is just np, the number of pade coefficients, i.e., 4. 
 
-  allocate(v(nz)); v=0.0_wp*v
+  allocate(v(nz)); v=cmplx(0.0_wp,0.0_wp,wp)
 
   nz=nz-2
   nz1=nz+1
```

### `src/envdata.f90` -- data module for new sediment variables

Added module-level variables to support the extended sediment model:

- `nzs` -- number of sediment depth points (was implicitly 4).
- `isedrd` -- flag: 0 = range-independent, 1 = range-dependent sediment.
- `nrp_sed` -- number of range-dependent sediment profiles.
- `rp_sed(:)` -- range points for sediment profiles (metres).
- `cs`, `rho`, `attn` changed from 1-D `(4)` to 2-D `(nzs, nrp_sed)`.

```diff
@@ -8,12 +8,21 @@
 implicit none
 
 real(kind=wp) :: sedlayer
+integer :: nzs                                             ! number of sediment depth points
 real(kind=wp),dimension(:),allocatable :: zg               ! depth grid with deltaz spacing
 real(kind=wp),dimension(:),allocatable :: rb,zb            ! bathymetry
 real(kind=wp),dimension(:),allocatable :: rp,zw            ! range and depths of sound speeds
 real(kind=wp),dimension(:,:),allocatable ::  cw            ! sound speeds, etc.
-real(kind=wp),dimension(:), allocatable :: cs,rho,attn    
-                           ! bottom properties are simple and range independent - four values.
+
+! Bottom sediment properties.
+! When isedrd==0 (range-independent): cs(nzs,1), rho(nzs,1), attn(nzs,1) -- single profile.
+! When isedrd==1 (range-dependent):   cs(nzs,nrp_sed), rho(nzs,nrp_sed), attn(nzs,nrp_sed)
+!   with rp_sed(nrp_sed) giving the range points (in metres).
+integer :: isedrd                                          ! 0=range-indep, 1=range-dep sediment
+integer :: nrp_sed                                         ! number of sediment range profiles
+real(kind=wp),dimension(:),   allocatable :: rp_sed        ! sediment range points (m)
+real(kind=wp),dimension(:,:), allocatable :: cs,rho,attn   ! bottom properties (nzs, nrp_sed)
 
  end module envdata
```

### `src/ram.f90` -- bug fixes and range-dependent sediment

#### NaN-safe initialisation (same pattern as matrc/solvetri)

```diff
@@ -96,8 +104,8 @@
   ! Self starter
   if (size(zsrc)==1) then
     allocate(uu(nz+2))
-    ! zero uu
-    uu=0.0_wp*uu
+    ! zero uu (use assignment, not multiply -- 0*NaN=NaN on uninitialized memory)
+    uu=cmplx(0.0_wp,0.0_wp,wp)
     ! Conditions for the delta function.
     zsc=1.0_wp+zsrc(1)/deltaz
     izs=floor(zsc)
@@ -106,8 +114,8 @@
     uu(izs+1)=      delzs* sqrt(2.0_wp*pi/k0)/(deltaz*alpw(izs))
 
     ! Divide the delta function by (1-X)**2 to get a smooth rhs.
-    allocate(pdu(np),pdl(np)); pdu=0.0_wp*pdu; pdl=0.0_wp*pdl
-    pdu(1)=cmplx(0.0_wp,wp); pdl(1)=cmplx(-1.0_wp,wp)
+    allocate(pdu(np),pdl(np)); pdu=cmplx(0.0_wp,0.0_wp,wp); pdl=cmplx(0.0_wp,0.0_wp,wp)
+    pdu(1)=cmplx(0.0_wp,0.0_wp,wp); pdl(1)=cmplx(-1.0_wp,0.0_wp,wp)
     call matrc
     call solvetri
     call solvetri
@@ -148,9 +156,9 @@
   if (size(zsrc)==size(zg)) uu=zsrc/f3
 
   if (.not.allocated(psi)) allocate(psi(nz+2,nr))
-    psi=0.0_wp*psi
+    psi=cmplx(0.0_wp,0.0_wp,wp)
   if (.not.allocated(rout)) allocate(rout(nr))
-    rout=0.0_wp*rout
+    rout=0.0_wp
```

#### Variable declarations for range-dependent sediment

```diff
@@ -35,8 +35,8 @@
 real(kind=wp),dimension(:),intent(in) :: zsrc,rg 
 
 integer :: iflag
-integer :: ii,n,nz,nr,nb,ir,irl,izl,izll,irr,upd,izs
-integer :: ir0(1)
+integer :: ii,n,nz,nr,nb,ir,irl,izl,izll,irr,upd,izs,ir_sed,irl_sed
+integer :: ir0(1), ir0_sed(1)
 real(kind=wp) :: omega, dr, rend, rnow, rint, rsc
 real(kind=wp) :: delzs,zbc,zsc
 real(kind=wp) :: rint1(1),zbc1(1), maxrb1
```

#### `rnow` initialisation fix

The original set `rnow = rg(1)` (first output range), which skipped PE
self-starter propagation from range zero.  The modified version keeps
`rnow = 0` so the field is correctly marched from the source.

```diff
@@ -59,7 +59,6 @@
   nr=size(rg)
   rend=rg(nr)
   rnow=0.0_wp
-  if (nr>1) rnow=rg(1)
 
   dr=deltar    ! dr is deltar of peramx; 
             ! it may need to adjust to get to the range rend precisely
```

#### Range-dependent sediment index tracking in march loop

```diff
@@ -76,6 +75,15 @@
   ir=ir0(1)
   irl=ir
 
+  ! Initialize sediment range index
+  if (isedrd==1 .and. nrp_sed>1) then
+     ir0_sed=minloc(abs(rp_sed-(rnow+dr/2.0_wp)))
+     ir_sed=ir0_sed(1)
+  else
+     ir_sed=1
+  end if
+  irl_sed=ir_sed
+
   nb=size(rb)
   allocate(rb1(nb+1),zb1(nb+1))
```

```diff
@@ -194,10 +202,21 @@
       ! Varying profiles - using profile closest to present range.
       ir0=minloc(abs(rp-rint))
       irl=ir; ir=ir0(1)
-      if (ir/=irl) then 
+      if (ir/=irl) then
       ! sound speed has changed; update profiles and call matrc
            iflag=iflag+2
-           upd=1    
+           upd=1
+      end if
+
+      ! Varying sediment profiles (range-dependent bottom)
+      if (isedrd==1 .and. nrp_sed>1) then
+         ir0_sed=minloc(abs(rp_sed-rint))
+         irl_sed=ir_sed; ir_sed=ir0_sed(1)
+         if (ir_sed/=irl_sed) then
+            ! sediment profile has changed; need to update bottom
+            if (mod(iflag,2)==0) iflag=iflag+1  ! ensure iflag includes 1 (bottom update)
+            upd=1
+         end if
       end if
```

#### `profl` subroutine -- configurable nzs-point sediment with range selection

The original used a hardcoded 4-point sediment model (surface, seafloor,
sedlayer depth, domain bottom) with range-independent properties.  The modified
version accepts `nzs` depth points and selects the nearest sediment profile by
range when `isedrd == 1`.

```diff
@@ -246,12 +265,12 @@
 integer, intent(in) :: iflag  ! iflag=3 update all; iflag=1 update bathymetry; iflag=2 update sound speed
 real(kind=wp), intent(in) :: r, omega
 
-integer :: ir,ii
-integer :: ir0(1)
-real(kind=wp) :: depth
-real(kind=wp), dimension(:) :: rwork(1),zwork(4)
+integer :: ir,ii,ir_sed,iz
+integer :: ir0(1), ir0_sed(1)
+real(kind=wp) :: depth, dz_sed
+real(kind=wp), dimension(:) :: rwork(1)
 real(kind=wp), dimension(:,:) :: work(1,1)
-real(kind=wp), dimension(:), allocatable :: csg,attng
+real(kind=wp), dimension(:), allocatable :: csg,attng,zwork,cs_local,rho_local,attn_local
 
   n=size(zg)
   if (.not.allocated(rhob)) allocate(cwg(n),rhob(n),ksqw(n),alpw(n),alpb(n),ksqb(n))
@@ -265,25 +284,44 @@
 
 if (iflag==1.or.iflag==3) then    ! update sediment sound speed, density, attenuation
     allocate(csg(n),attng(n))
+    allocate(zwork(nzs),cs_local(nzs),rho_local(nzs),attn_local(nzs))
+
+!   Select sediment profile for this range
+    if (isedrd==1 .and. nrp_sed>1) then
+       ir0_sed=minloc(abs(rp_sed-r)); ir_sed=ir0_sed(1)
+    else
+       ir_sed=1
+    end if
+    cs_local   = cs(:,ir_sed)
+    rho_local  = rho(:,ir_sed)
+    attn_local = attn(:,ir_sed)
 
 !   First find the depth at this range
     rwork(1)=r
     work(:,1)=interp1(rb,zb,rwork,zb(1))
     depth=work(1,1)
-    ! The four values of depth that go with cs, rho, and attn
-    zwork(1)=0.0_wp; zwork(2)=depth
-    zwork(3)=depth+sedlayer; zwork(4)=max(zg(n),zwork(3)+1.0E-6)
-
-! Set up sediment sound speed to increase linearly below the sea floor, with
-! a sedlayer-m thick sediment layer. 
-    csg=gorp(zwork,cs,zg)
+
+!   Construct nzs depth points: surface, seafloor, nzs-3 interior sediment
+!   points evenly spaced, then domain bottom.
+    zwork(1)=0.0_wp
+    zwork(2)=depth
+    if (nzs > 3) then
+       dz_sed = sedlayer / real(nzs-3, wp)
+       do iz=3,nzs-1
+          zwork(iz) = depth + real(iz-2, wp) * dz_sed
+       end do
+    end if
+    zwork(nzs) = max(zg(n), zwork(nzs-1)+1.0E-6_wp)
+
+! Set up sediment sound speed profile (linearly interpolated over nzs points).
+    csg=gorp(zwork,cs_local,zg)
     csg=cwg+csg
 
 ! Set up the sediment density and attenuation profiles.
-! Attenuation and density follow the bottom.
-    rwork(1)=rho(1) 
-    rhob=gorp(zwork,rwork,zg)   ! send it only one value, so gorp does the easy thing; rhob a constant.
-    attng=gorp(zwork,attn,zg)
+    rhob=gorp(zwork,rho_local,zg)
+    attng=gorp(zwork,attn_local,zg)
+
+    deallocate(zwork,cs_local,rho_local,attn_local)
  end if
```

#### `gorp` function -- NaN-safe broadcast

```diff
@@ -323,7 +361,7 @@
   
   if (size(y)==1) then
     !forall(ii=1:n) gorp(ii)=y(1)
-    gorp=y(1)+0.0_wp*gorp
+    gorp=y(1)
     return
   end if
```

### `src/peramx.f90` -- I/O rewrite (largest change)

#### Free-format input parsing and longer filename buffers

The original fixed-format reads (`read(2,'(f4.0)')`) required values to fit in
exact column widths.  Free-format `read(nunit,*)` is standard practice for
program-controlled input files.  Filename buffers increased from 20 to 256
characters to accommodate full paths.

```diff
@@ -22,19 +22,19 @@
 
 real(kind=wp),dimension(:),allocatable :: zg1
 complex(kind=wp),dimension(:),allocatable :: psi1
-complex(kind=wp),dimension(:,:),allocatable :: psif
+complex(kind=wp),dimension(:,:,:),allocatable :: psif  ! (nzo, nf, nr)
 
 ! input parameters - c.f., file "in.pe"
 integer :: dzm, iflat, ihorz, ibot
 real(kind=wp) :: fc,Q,T,dum
 real(kind=wp),dimension(:),allocatable :: zsrc,rmax
-character(len=20) :: name1,name2     ! sound speed and bathymetry filenames
+character(len=256) :: name1,name2,name3,name4  ! ssp, bathymetry, ranges, sediment filenames
 real(kind=wp),dimension(:),allocatable :: eps
 real(kind=wp),dimension(:,:),allocatable :: cq
 
 integer :: nss
 
-integer :: nb,nzp,nrp,nrp0,n,nf1,nf
+integer :: nb,nzp,nrp,nrp0,n,nf1,nf,nr
 real(kind=wp) :: bw, fs, Nsam, df, tmp
 real(kind=wp),dimension(:),allocatable :: frq
 
@@ -44,10 +44,10 @@
 real(kind=wp) :: rate
 integer :: t1,t2,cr,cm
 
-integer :: ii,jj,iff,length
+integer :: ii,jj,iff,ir,length
 
 integer, parameter :: nunit=2
-complex(kind=wp), parameter :: j=cmplx(0.0_wp,1.0_wp)
+complex(kind=wp), parameter :: j=cmplx(0.0_wp,1.0_wp,wp)
 complex(kind=wp) :: scl
```

```diff
-allocate(zsrc(1),rmax(1))
+allocate(zsrc(1))
 
 open(nunit,file='in.pe',status='old')
-read (2,'(f4.0)')  fc                ! skip the first line - read a dummy
-read (2,'(f4.0,1x,f2.0)') fc,Q       ! center frequency and Q
-read (2,'(f4.1)') T              ! time window width
-read (2,'(f6.1)') zsrc(1)        ! source depth
-read (2,'(f12.3)') rmax(1)       ! receiver range
-read (2,'(f5.2)') deltaz         ! depth accuracy parameter
-read (2,'(f6.2)') deltar         ! range accuracy parameter
-read (2,'(i1,1x,i1)') np,nss     ! np -# pade coefficients
-                                 ! ns -# stability terms
-read (2,'(f7.1)') rs             ! stability range
-read (2,'(i2)') dzm              ! dzm - depth decimation
-read (2,'(a20)') name1           ! sound speed filename; "munk" will just use canonical
-name1=trim(name1) ! remove trailing blanks
-read (2,'(i1)') iflat            ! 0=no flat earth transform, 1=yes
-read (2,'(i1)') ihorz            ! 0=no horizontal linear interpolation, 1=yes
-read (2,'(i1)') ibot             ! 0=no bottom, 1=bottom and read a file
-read (2,'(a20)') name2           ! bathymetry filename; ignored if ibot=0
-name2=trim(name2) ! remove trailing blanks
+read (nunit,*) dum                    ! skip the first line (comment/title)
+read (nunit,*) fc, Q                  ! center frequency (Hz) and Q value
+read (nunit,*) T                      ! time window width (s)
+read (nunit,*) zsrc(1)                ! source depth (m)
+read (nunit,*) deltaz                 ! depth accuracy parameter (m)
+read (nunit,*) deltar                 ! range accuracy parameter (m)
+read (nunit,*) np, nss                ! np-# pade coefficients, ns-# stability terms
+read (nunit,*) rs                     ! stability range (m)
+read (nunit,*) dzm                    ! output depth decimation (integer)
+read (nunit,'(a)') name1              ! sound speed filename
+name1=trim(adjustl(name1))
+read (nunit,*) iflat                  ! 0=no flat earth transform, 1=yes
+read (nunit,*) ihorz                  ! 0=no horizontal linear interpolation, 1=yes
+read (nunit,*) ibot                   ! 0=no bottom, 1=bottom and read a file
+read (nunit,'(a)') name2              ! bathymetry filename
+name2=trim(adjustl(name2))
+read (nunit,'(a)') name3              ! output ranges filename
+name3=trim(adjustl(name3))
```

#### Configurable sediment properties (read from `in.pe` instead of hardcoded)

Note: the original `rmax` line (`read (2,'(f12.3)') rmax(1)`) is removed from
the input parsing block above -- it is replaced by the external ranges file
(see "Multiple output ranges" below).

`sedlayer`, `nzs`, `cs`, `rho`, `attn` and the range-dependent sediment flag
`isedrd` are now read from `in.pe`.  Supports an optional external sediment
profile file when `isedrd == 1`.

```diff
+! Read bottom properties (sedlayer, nzs, cs, rho, attn)
+read (nunit,*) sedlayer
+read (nunit,*) nzs
+read (nunit,*) isedrd
+
+if (isedrd==1) then
+   ! Range-dependent sediment: read filename and load profiles
+   read (nunit,'(a)') name4
+   name4=trim(adjustl(name4))
+
+   ! Temporary defaults (overridden by file)
+   nrp_sed=1
+   allocate(cs(nzs,1),rho(nzs,1),attn(nzs,1))
+   cs(:,1)  = 0.0_wp
+   rho(:,1) = 1.2_wp
+   attn(:,1)= 0.5_wp
+
+   close(nunit)
+
+   ! Read sediment profile file (same format as SSP: "-1 range_km" headers)
+   print *,'Reading sediment file: ', trim(name4)
+   open(nunit,file=name4,status='old')
+
+   ! First pass: count profiles
+   deallocate(cs,rho,attn)
+   nrp_sed=0
+   do
+      read(nunit,*,end=4) dum
+      if (dum<0) nrp_sed=nrp_sed+1
+   end do
+4  print *,'Found ',nrp_sed,' sediment profiles.'
+   rewind(nunit)
+
+   allocate(rp_sed(nrp_sed), cs(nzs,nrp_sed), rho(nzs,nrp_sed), attn(nzs,nrp_sed))
+
+   ! Second pass: read profiles (nzs values per line)
+   do ii=1,nrp_sed
+      read(nunit,*) dum, rp_sed(ii)
+      rp_sed(ii) = rp_sed(ii)*1000.0_wp   ! convert km to m
+      read(nunit,*) (cs(jj,ii), jj=1,nzs)
+      read(nunit,*) (rho(jj,ii), jj=1,nzs)
+      read(nunit,*) (attn(jj,ii), jj=1,nzs)
+   end do
+   close(nunit)
+
+else
+   ! Range-independent sediment: read nzs-element arrays from in.pe
+   nrp_sed=1
+   allocate(cs(nzs,1),rho(nzs,1),attn(nzs,1))
+   read (nunit,*) (cs(jj,1), jj=1,nzs)
+   read (nunit,*) (rho(jj,1), jj=1,nzs)
+   read (nunit,*) (attn(jj,1), jj=1,nzs)
+   close(nunit)
+end if
```

Replaces the original hardcoded bottom properties block:

```diff
-! Sediment layer thickness - meters
-sedlayer=300.0_wp
-
-! Sediment sound speed - this will be speed relative to the water sound speed.
-! Four values are given:  at the surface, at the bottom, sedlayer-m below the bottom,
-! and at the center of the earth...
-allocate(cs(4))
-cs(1)=0.0_wp   
-cs(2)=0.0_wp
-cs(3)=200.0_wp 
-cs(4)=200.0_wp
-
-allocate(rho(4))
-rho(1)=1.2_wp 
-rho(2)=1.2_wp
-rho(3)=1.2_wp 
-rho(4)=1.2_wp
-
-allocate(attn(4))
-attn(1)=0.5_wp  
-attn(2)=0.5_wp
-attn(3)=5.0_wp  
-attn(4)=5.0_wp
+! Bottom properties (sedlayer, nzs, cs, rho, attn) were read from in.pe above.
+! cs is sediment speed relative to the water sound speed (nzs values:
+!   at surface, at seafloor, evenly spaced through sediment, at domain bottom).
+! rho is sediment density (g/cm^3), nzs values.
+! attn is sediment attenuation (dB/wavelength), nzs values.
```

#### Multiple output ranges from external file

The original computed the field at a single range.  The modified version reads
an arbitrary list of output ranges from an external file and stores the field
at each one.  `psif` is now 3-D `(nzo, nf, nr)`.

```diff
+! Read output ranges from file
+print *,'Reading output ranges file: ', trim(name3)
+open(nunit,file=name3,status='old')
+nr=0
+do
+   read(nunit,*,end=6) dum
+   nr=nr+1
+end do
+6 print *,'Found ',nr,' output ranges.'
+rewind(nunit)
+allocate(rmax(nr))
+do ii=1,nr
+   read(nunit,*) rmax(ii)
+end do
+close(nunit)
```

#### Updated diagnostic print statements

Format widths adjusted for new parameter ranges, multi-range output info added,
sediment properties echoed.  Debug print for `c0`/`cmin` added.

```diff
@@ -67 (continued: print block)
 print '(a)','INPUT PARAMETERS:'
 print '(a,f10.2)','Center frequency (Hz): ', fc
-print '(a,f2.0)','Q: ', Q
+print '(a,f4.1)','Q: ', Q
 print '(a,f5.2)','Bandwidth (f0/Q - Hz): ', fc/Q
-print '(a,f4.1)','Time window width (s): ', T
-print '(a,f6.1)','Source depth (m): ', zsrc(1)
-print '(a,f12.3)','Range (m): ', rmax(1)
+print '(a,f6.1)','Time window width (s): ', T
+print '(a,f8.1)','Source depth (m): ', zsrc(1)
+print '(a,i6)','Number of output ranges: ', nr
+print '(a,f12.1)','First range (m): ', rmax(1)
+print '(a,f12.1)','Last range (m): ', rmax(nr)
 ...
-print '(a,i2)','Output depth decimation: ', dzm
-print '(a,a)','Sound speed filename: ', name1
+print '(a,i4)','Output depth decimation: ', dzm
+print '(a,a)','Sound speed filename: ', trim(name1)
 ...
-print '(a,a)','Ocean bottom filename: ', name2
+print '(a,a)','Ocean bottom filename: ', trim(name2)
+print '(a,a)','Ranges filename: ', trim(name3)
+print '(a,f8.1)','Sediment layer (m): ', sedlayer
+print '(a,i4)','Sediment depth points (nzs): ', nzs
+if (isedrd==1) then
+   print '(a,i4,a)','Sediment: range-dependent (',nrp_sed,' profiles)'
+else
+   print '(a,*(f8.2))','Sediment speed (cs): ', cs(:,1)
+   print '(a,*(f8.3))','Sediment density (rho): ', rho(:,1)
+   print '(a,*(f8.3))','Sediment attenuation: ', attn(:,1)
+end if
```

```diff
@@ -196,6 +277,7 @@
 c0=sum(cw)/n
 ic0=1.0_wp/c0
 cmin=minval(cw)     ! minimum sound speed for calculating tdelay
+print '(a,f10.2,a,f10.2)', 'c0=',c0,' cmin=',cmin
```

#### Horizontal interpolation range fix

The original used `rmax(1)` (single range) for the SSP horizontal
interpolation grid.  Now uses `maxval(rmax)` to span all output ranges.

```diff
@@ -159,8 +240,8 @@
    
    if (ihorz==1) then
    ! Horizontal
-     nrp=nint(rmax(1)/10000.0_wp)
-     call linspace(rp, rp0(1),rmax(1),nrp)
+     nrp=nint(maxval(rmax)/10000.0_wp)
+     call linspace(rp, rp0(1),maxval(rmax),nrp)
      allocate(cw(nzp,nrp))
      do jj=1,nzp
         cw(jj,:)=interp1(rp0,cq(jj,:),rp,cq(jj,1))
```

#### OpenMP race condition fix and multi-range parallel loop

`zg` (the depth grid) is pre-allocated before the `!$OMP PARALLEL` region.
Without this, multiple threads entering `ram()` could simultaneously find `zg`
unallocated and race on allocation.  The inner loop now iterates over output
ranges.

```diff
-allocate(psif(nzo,nf))
+allocate(psif(nzo,nf,nr))
+
+! Pre-allocate zg before the parallel region to avoid an OpenMP race
+! condition: without this, multiple threads entering ram() could
+! simultaneously see zg as unallocated and both try to allocate it.
+call linspace(zg, 0.0_wp, zmax, icount)
 
  call system_clock(count_rate=cr)
  call system_clock(count_max=cm)
  rate=real(cr)
 
-print *,nf,' total frequencies'
+print *,nf,' total frequencies, ',nr,' output ranges'
```

```diff
-!$OMP PARALLEL PRIVATE (psi1,omega,scl,t1,t2,cr,rate) 
+!$OMP PARALLEL PRIVATE (psi1,omega,scl,t1,t2,cr,rate,ir)
 allocate(psi1(nzo))
 !$OMP DO SCHEDULE(STATIC,1)
   do iff=1,nf
@@ -345,16 +397,13 @@
 
     call ram(zsrc,rmax)
 
-! The miracle of fortran95!
-    psi1=psi(1:icount:dzm,1)
-
-    omega=2.0_wp*pi*frqq 
-    ! 3-D
-    scl=exp(j*(omega/c0*rout(1) + pi/4.0_wp))/4.0_wp/pi
-    ! 2-D
-    ! k0=omega/c0
-    !scl=j*exp(j*omega/c0*rout)/sqrt(8.0_wp*pi*k0)
-    psif(:,iff)=scl*psi1
+    omega=2.0_wp*pi*frqq
+    do ir=1,nr
+      psi1=psi(1:icount:dzm,ir)
+      ! 3-D scaling
+      scl=exp(j*(omega/c0*rout(ir) + pi/4.0_wp))/4.0_wp/pi
+      psif(:,iff,ir)=scl*psi1
+    end do
```

#### Flat-earth inverse-transform guard

The original applied the inverse flat-earth depth correction unconditionally at
output time, even when `iflat=0` (no forward transform was applied).  The
modified version wraps it in `if (iflat==1)`.

```diff
-! Remove the flat-earth transform (or most of it, anyways)
-allocate(eps(nzo))
-eps=zg1*invRe
-zg1=zg1/(1.0_wp+(1.0_wp/2.0_wp)*eps+(1.0_wp/3.0_wp)*eps*eps)
-deallocate(eps)
+!  Remove the flat-earth transform (or most of it, anyways)
+  if (iflat==1) then
+    allocate(eps(nzo))
+    eps=zg1*invRe
+    zg1=zg1/(1.0_wp+(1.0_wp/2.0_wp)*eps+(1.0_wp/3.0_wp)*eps*eps)
+    deallocate(eps)
+  end if
```

#### Output file format rewrite

`psif.dat` header now includes the number of output ranges `nr` and a dedicated
record for `rout(1:nr)`.  Data records loop over ranges.  The record-length
calculation uses `inquire(iolength=...)` instead of manual arithmetic.  The
Python reader `io/mpirams_reader.py` matches this format.

```diff
-inquire(iolength=length) real(psif(1,:))
-length=2*length+length/nf  ! We need nf real and nf imaginary values and the depth.
-! length is the record length, an important piece of information
-! Write it out for handy reference:
+block
+  integer :: rl1
+  inquire(iolength=rl1) fc        ! iolength of one real(wp)
+  length = max(8, nf, nr, 1+2*nf) * rl1
+end block
+
 open(nunit, form='formatted',file='recl.dat')
 write(nunit,*) length
 close(nunit)
 
-! Now write out the data to a direct access file:
 open(nunit, access='direct',recl=length,file='psif.dat')
 
-! Float the integers to real, otherwise there will be trouble.
-write(nunit,rec=1) Nsam,real(nf,wp),real(nzo,wp),rout,c0,cmin,fs,Q
-write(nunit,rec=2) frq     ! vector of size nf
-
-do ii=1,nzo
- write(nunit,rec=ii+2) zg1(ii),((real(psif(ii,jj))),(aimag(psif(ii,jj))),jj=1,nf)
+! Record 1: header parameters
+write(nunit,rec=1) Nsam,real(nf,wp),real(nzo,wp),real(nr,wp),c0,cmin,fs,Q
+! Record 2: frequency vector
+write(nunit,rec=2) frq
+! Record 3: output ranges
+write(nunit,rec=3) rout
+
+! Records 4+: data blocks, one per range, each containing nzo depth records
+do ir=1,nr
+  do ii=1,nzo
+    write(nunit,rec=3+(ir-1)*nzo+ii) zg1(ii), &
+        ((real(psif(ii,jj,ir))),(aimag(psif(ii,jj,ir))),jj=1,nf)
+  end do
  end do
 
  close(nunit)
 
  print *, 'ALL DONE! '
+print '(a,i6,a,i4,a)','Wrote ',nzo,' depths x ',nr,' ranges to psif.dat'
  print *,'recl.dat has the record length of the direct access file.'
-print *,'The direct access file with the parameters and results is psif.dat.'
```
