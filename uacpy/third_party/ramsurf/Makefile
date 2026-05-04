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
FFLAGS  ?= -O2 -std=legacy -w
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
