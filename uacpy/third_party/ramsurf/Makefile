# Build the Collins-style RAM-family PE codes uacpy actually dispatches to —
# rams0.5 for elastic bottoms, ramsurf1.5 for rough surfaces. Skips:
#   - ram1.5.f: original fluid PE; uacpy uses mpiramS instead (broadband,
#     better numerics, range-dependent layered bottom support). The .f
#     source stays vendored for reference but no binary is built.
#   - ramsurf.c, ramsurflib.c: C reimplementation, fluid-only.
#   - ramclr.f, ramsurfclr2.0.f: PostScript plotters (uacpy plots via
#     matplotlib).
# Bypasses the upstream autotools setup (configure.ac / Makefile.am) so we
# don't need autoconf/automake on build hosts.

FC      ?= gfortran
# -fdefault-real-8/-fdefault-double-8: the Collins march outside epade is
# single precision (ram.pdf section 3), which makes refining dz past ~lambda/120
# lose more to round-off than the finer grid gains. epade's explicit
# complex*8/real*4 declarations are promoted in the source to match.
FFLAGS  ?= -O2 -std=legacy -w -fdefault-real-8 -fdefault-double-8
LDFLAGS ?=

BINS = rams0.5 ramsurf1.5

all: $(BINS)

rams0.5: rams0.5.f
	$(FC) $(FFLAGS) -o $@ $< $(LDFLAGS)

ramsurf1.5: ramsurf1.5.f
	$(FC) $(FFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(BINS)

.PHONY: all clean
