# RAMGEO — range-dependent acoustic model, geoacoustic (layered) bottom

`ramgeo1.5.f` is Michael D. Collins' **RAMGEO**, the variant of the
Range-dependent Acoustic Model (split-step Padé parabolic equation) that
*"handles multiple sediment layers that parallel the bathymetry"* — i.e. a
**range-dependent layered fluid** seabed.

- **Author / origin:** Michael D. Collins, Naval Research Laboratory
  (Washington, DC). Originally distributed via `ram.nrl.navy.mil`.
- **Source:** the **Acoustics Toolbox** `RAM/` bundle (Porter's AT, mirroring
  `oalib.hlsresearch.com/Modes/AcousticsToolbox/`); `ramgeo1.5.f` is byte-for-byte
  that copy plus uacpy's two patches.
- **Licence:** **public domain** — a U.S. Government work (Collins, NRL). No
  explicit licence accompanies the code: NRL/OALIB distribute it freely as
  "M. Collins, NRL, 1999" and the source carries no copyright or licence notice.
  Porter merely bundles it in the Acoustics Toolbox; bundling does not
  relicense it.
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
