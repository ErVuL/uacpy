# Build RAMGEO (ramgeo1.5) — Collins' range-dependent acoustic model with
# multiple fluid sediment layers that parallel the bathymetry. Reads
# `ramgeo.in`, writes `tl.line` (text) and `tl.grid` (unformatted), the same
# output family as rams0.5 / ramsurf1.5.
#
# Single source, no autotools — keeps build hosts free of autoconf/automake.

FC      ?= gfortran
FFLAGS  ?= -O2 -std=legacy -w
LDFLAGS ?=

BINS = ramgeo

all: $(BINS)

ramgeo: ramgeo1.5.f
	$(FC) $(FFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(BINS)

.PHONY: all clean
