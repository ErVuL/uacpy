PROG = s_mpiram
# Makefile for single processor version.  

OBJDIR = obj
MODDIR = mod
SRCDIR = src


###########################################
# Gnu g77/gfortran options (64 bit)
FC = gfortran

#FFLAGS = -march=native -mtune=native -fopenmp -m64 -mfpmath=sse -I $(MODDIR) -Wall -finline-functions -ffast-math -fno-strength-reduce -falign-functions=2  -O3 -fomit-frame-pointer 
#FFLAGS = -g -pg -march=native -fopenmp -m64 -mfpmath=sse -I $(MODDIR) -Wall 
#FFLAGS = -march=corei7-avx -mtune=corei7-avx -fopenmp -m64 -mfpmath=sse -I $(MODDIR) -Wall -finline-functions -ffast-math -fno-strength-reduce -fomit-frame-pointer -falign-functions=2  -O3 -fuse-linker-plugin
#FFLAGS = -march=native -mtune=native -fopenmp -m64 -mfpmath=sse -I $(MODDIR) -Wall -finline-functions -ffast-math -fno-strength-reduce -fomit-frame-pointer -falign-functions=2  -O3 

#LDFLAGS = -fopenmp -march=corei7-avx -mtune=corei7-avx -flto
#LDFLAGS = -fopenmp -march=native -mtune=native
#LDFLAGS = -g -pg -fopenmp -march=native -mtune=native 

FFLAGS = -Ofast -march=native -fopenmp -I $(MODDIR) -Wall -fuse-linker-plugin
LDFLAGS = -Ofast -fopenmp -march=native -flto

#####################################################
# The Nvidia compiler 
#FC = nvfortran
#FFLAGS = -Ofast -mp -tp=icelake -Mcache_align -module $(MODDIR)
#LDFLAGS = -Ofast -mp -tp=icelake -Mcache_align 

OBJ    = 			\
	$(OBJDIR)/kinds.o	\
	$(OBJDIR)/envdata.o	\
	$(OBJDIR)/param.o	\
	$(OBJDIR)/mattri.o	\
	$(OBJDIR)/interpolators.o	\
	$(OBJDIR)/profiles.o	\
	$(OBJDIR)/fld.o	\
	$(OBJDIR)/epade.o	\
	$(OBJDIR)/matrc.o	\
	$(OBJDIR)/peramx.o	\
	$(OBJDIR)/ram.o	\
	$(OBJDIR)/solvetri.o	\
	$(OBJDIR)/splnlib.o	

MOD    = 			\
	$(MODDIR)/kinds.mod	\
	$(MODDIR)/envdata.mod	\
	$(MODDIR)/param.mod	\
	$(MODDIR)/mattri.mod	\
	$(MODDIR)/interpolators.mod	\
	$(MODDIR)/profiles.mod	\
	$(MODDIR)/fld.mod	

$(PROG) : $(MOD) $(OBJ) 
	$(FC) -o $(PROG) $(OBJ) $(LDFLAGS)
#	strip $(PROG)

clean: 
	rm -f obj/peramx_mpi.o
	rm -f $(OBJ) $(MOD) 

# First compile the modules - these are needed when code using them are compiled.
$(MODDIR)/kinds.mod : $(SRCDIR)/kinds.f90
	$(FC) -c $(FFLAGS)  $(SRCDIR)/kinds.f90
	mv -f kinds.mod $(MODDIR)/
	mv -f kinds.o $(OBJDIR)/

$(MODDIR)/envdata.mod : $(SRCDIR)/envdata.f90
	$(FC) -c $(FFLAGS)  $(SRCDIR)/envdata.f90
	mv -f envdata.mod $(MODDIR)/
	mv -f envdata.o $(OBJDIR)/

$(MODDIR)/param.mod : $(SRCDIR)/param.f90
	$(FC) -c $(FFLAGS)  $(SRCDIR)/param.f90
	mv -f param.mod $(MODDIR)/
	mv -f param.o $(OBJDIR)/

$(MODDIR)/mattri.mod : $(SRCDIR)/mattri.f90
	$(FC) -c $(FFLAGS)  $(SRCDIR)/mattri.f90
	mv -f mattri.mod $(MODDIR)/
	mv -f mattri.o $(OBJDIR)/

$(MODDIR)/interpolators.mod : $(SRCDIR)/interpolators.f90
	$(FC) -c $(FFLAGS)  $(SRCDIR)/interpolators.f90
	mv -f interpolators.mod $(MODDIR)/
	mv -f interpolators.o $(OBJDIR)/

$(MODDIR)/profiles.mod : $(SRCDIR)/profiles.f90
	$(FC) -c $(FFLAGS)  $(SRCDIR)/profiles.f90
	mv -f profiles.mod $(MODDIR)/
	mv -f profiles.o $(OBJDIR)/

$(MODDIR)/fld.mod : $(SRCDIR)/fld.f90
	$(FC) -c $(FFLAGS)  $(SRCDIR)/fld.f90
	mv -f fld.mod $(MODDIR)/
	mv -f fld.o $(OBJDIR)/

# Object files (nothing to be done for the modules.)
$(OBJDIR)/kinds.o : $(SRCDIR)/kinds.f90 $(MOD)

$(OBJDIR)/envdata.o : $(SRCDIR)/envdata.f90 $(MOD)

$(OBJDIR)/param.o : $(SRCDIR)/param.f90 $(MOD)

$(OBJDIR)/mattri.o : $(SRCDIR)/mattri.f90 $(MOD)

$(OBJDIR)/interpolators.o : $(SRCDIR)/interpolators.f90  $(MOD)

$(OBJDIR)/profiles.o : $(SRCDIR)/profiles.f90 $(MOD)

$(OBJDIR)/fld.o : $(SRCDIR)/fld.f90 $(MOD)

$(OBJDIR)/epade.o : $(SRCDIR)/epade.f90 $(MOD)
	$(FC) -c $(FFLAGS)  $(SRCDIR)/epade.f90
	mv -f epade.o $(OBJDIR)/

$(OBJDIR)/matrc.o : $(SRCDIR)/matrc.f90 $(MOD)
	$(FC) -c $(FFLAGS)  $(SRCDIR)/matrc.f90
	mv -f matrc.o $(OBJDIR)/

$(OBJDIR)/peramx.o : $(SRCDIR)/peramx.f90 $(MOD)
	$(FC) -c $(FFLAGS)  $(SRCDIR)/peramx.f90
	mv -f peramx.o $(OBJDIR)/

$(OBJDIR)/ram.o : $(SRCDIR)/ram.f90 $(MOD)
	$(FC) -c $(FFLAGS)  $(SRCDIR)/ram.f90
	mv -f ram.o $(OBJDIR)/

$(OBJDIR)/solvetri.o : $(SRCDIR)/solvetri.f90 $(MOD)
	$(FC) -c $(FFLAGS)  $(SRCDIR)/solvetri.f90
	mv -f solvetri.o $(OBJDIR)/

$(OBJDIR)/splnlib.o : $(SRCDIR)/splnlib.f90 $(MOD)
	$(FC) -c $(FFLAGS)  $(SRCDIR)/splnlib.f90
	mv -f splnlib.o $(OBJDIR)/

