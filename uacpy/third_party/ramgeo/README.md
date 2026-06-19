# RAMGEO — range-dependent acoustic model, geoacoustic (layered) bottom

`ramgeo1.5.f` is Michael D. Collins' **RAMGEO**, the variant of the
Range-dependent Acoustic Model (split-step Padé parabolic equation) that
*"handles multiple sediment layers that parallel the bathymetry"* — i.e. a
**range-dependent layered fluid** seabed.

- **Author / origin:** Michael D. Collins, U.S. Naval Research Laboratory
  (Washington, DC). Originally distributed via `ram.nrl.navy.mil`.
- **Licence:** U.S. Government work — **public domain**. No usage restriction
  and nothing extra to redistribute (cf. the BSD-3 quiet-oceans `ramsurf/`
  port). Sourced from the OALIB Acoustics Toolbox mirror.
- **uacpy patches:** two, both documented in `third_party/MODIFICATIONS.md`
  (the same pair `ramsurf1.5.f` carries): (1) enlarged array dimensions
  `(mr,mz,mp)` so uacpy's fine grids fit; (2) an `outpt` dump of the complex
  PE envelope to `pcomplex.bin`, enabling broadband / time-series synthesis.
- **Numerics:** split-step Padé PE, Collins, *J. Acoust. Soc. Am.* **93**,
  1736–1742 (1993). Version 1.5g (Fialkowski dimension-bug fix).

## Input / output

- Reads `ramgeo.in` (Collins text format: `freq zs zr` / `rmax dr ndr` /
  `zmax dz ndz zmplt` / `c0 np ns rs`, then `-1 -1`-terminated blocks for
  bathymetry `(rb, zb)`, water SSP `(z, cw)`, and the bottom profiles
  `(z, cb)`, `(z, rhob)`, `(z, attn)`; an optional bare range line starts a
  new profile section, the mechanism for range dependence).
- Writes `tl.line` (text, TL at `zr`) and `tl.grid` (unformatted, depth ×
  range), the same outputs as `rams0.5` / `ramsurf1.5`.

`ramgeo.in` here is the upstream sample (a range-dependent, layered fluid
case) kept for a smoke test.

## Relationship to the other RAM backends

uacpy's RAM dispatcher auto-selects RAMGEO for a **narrowband (COHERENT_TL),
fluid, flat-surface** environment whose bottom is **layered** — the case
where its bathymetry-parallel sediment layers are the most faithful Collins
treatment. `mpiramS` keeps the broadband fluid path and the non-layered
cases; `rams0.5` handles elastic bottoms, `ramsurf1.5` rough surfaces.
`RAM(backend='ramgeo')` forces it for any mode.
