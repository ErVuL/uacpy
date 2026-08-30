# Build RAMGEO (ramgeo1.5) — Collins' range-dependent acoustic model with
# multiple fluid sediment layers that parallel the bathymetry. Reads
# `ramgeo.in`, writes `tl.line` (text) and `tl.grid` (unformatted), the same
# output family as rams0.5 / ramsurf1.5.
#
# Single source, no autotools — keeps build hosts free of autoconf/automake.

FC      ?= gfortran
# -fdefault-real-8/-fdefault-double-8: the Collins march outside epade is
# single precision (ram.pdf section 3), which makes refining dz past ~lambda/120
# lose more to round-off than the finer grid gains. epade's explicit
# complex*8/real*4 declarations are promoted in the source to match.
FFLAGS  ?= -O2 -std=legacy -w -fdefault-real-8 -fdefault-double-8
LDFLAGS ?=

BINS = ramgeo

all: $(BINS)

ramgeo: ramgeo1.5.f
	$(FC) $(FFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(BINS)

.PHONY: all clean
