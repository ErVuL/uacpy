"""Geoacoustic property presets for common seafloor materials.

Class-typical compressional / shear / density / attenuation values for
the ocean-bottom material classes that appear in ocean-acoustics
modelling. Site-specific surveys should override individual fields.

Each preset captures every layer property the rest of uacpy can use:

============= ================================================
Key           Meaning
============= ================================================
sound_speed   Compressional wave speed ``c_p`` (m/s).
density       Mass density (g/cm³).
attenuation   Compressional attenuation ``α_p`` (dB/λ_p).
shear_speed   Shear wave speed ``c_s`` (m/s); 0 marks a fluid sediment.
shear_attenuation
              Shear attenuation ``α_s`` (dB/λ_s).
porosity      Volume fraction of pore water (%); ``None`` for rocks.
grain_size_phi
              Mean grain size on the Wentworth ϕ scale (informational metadata,
              and the input to ``BoundaryProperties.from_grain_size``);
              ``None`` for consolidated rocks where ϕ is not defined. Gravel's
              −1.0 is inside ``model='apl-uw'``'s fitted range and outside the
              default ``'hamilton'``'s, which answers it with its coarse-sand
              end row (see :func:`uacpy.core.sediment.grain_size_to_geoacoustics`).
roughness     RMS interface roughness (m); 0 unless overridden.
============= ================================================

``c_s`` for silt, sand and gravel is a depth-dependent ``c_s(z̄)`` in the source
table rather than a number — JKPS §1.6: the shear speeds of unconsolidated
sediments "are quite low but increase rapidly with depth z̄ below the
water-bottom interface". Those three presets carry one value each in its place
and the depth it stands for is not recorded in the table, so pass an explicit
``shear_speed`` whenever the depth matters.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from uacpy.core.exceptions import ConfigurationError


def _entry(
    *,
    sound_speed: float,
    density: float,
    attenuation: float,
    shear_speed: float = 0.0,
    shear_attenuation: float = 0.0,
    porosity: Optional[float] = None,
    grain_size_phi: Optional[float] = None,
    roughness: float = 0.0,
) -> Dict:
    return dict(
        sound_speed=float(sound_speed),
        density=float(density),
        attenuation=float(attenuation),
        shear_speed=float(shear_speed),
        shear_attenuation=float(shear_attenuation),
        porosity=porosity,
        grain_size_phi=grain_size_phi,
        roughness=float(roughness),
    )


# Wentworth classes, for orientation: clay ϕ ≥ 8, silt 4–8, very fine sand 3–4,
# fine sand 2–3, medium sand 1–2, coarse sand 0–1, very coarse sand −1–0,
# gravel −8..−1. The sand and silt grades are TR 9407 Table 2's Mz read as
# class midpoints, p. IV-7 giving the rule: for "the Wentworth and Lane
# schemes, the M_z value given in Table 2 is the midpoint of the range defined
# by the sediment name". Gravel is Medwin & Clay, *Fundamentals of Acoustical
# Oceanography*: "Marine geologists define gravel as being the loose material
# that ranges in size from 2 to 256 mm", i.e. −1 to −8 ϕ — seven ϕ wide, so
# gravel is a range of classes and not one mean grain size, and TR 9407
# Table 2 accordingly gives its "Cobble, Gravel, Pebble" row no Mz at all.
#
# Provenance, by row and by column. Four sources feed this table and no row is
# a blend of two of them:
#   c_p, density, alpha_p, c_s, alpha_s, porosity, rows clay … basalt
#                                        -> JKPS Table 1.3
#   every value in the granite row       -> Ainslie (2010) Table 4.20
#   grain_size_phi for clay, silt, sand  -> Hamilton & Bachman (1982) Table I
#   grain_size_phi for gravel            -> APL-UW TR 9407 Table 2
# Each is given in full below, with what it does and does not cover.
#
# c_p, density, alpha_p, c_s, alpha_s and porosity for the eight rows clay …
# basalt are Jensen, Kuperman, Porter & Schmidt, *Computational Ocean
# Acoustics*, Table 1.3 (continental shelf and slope), which carries no grain
# size. Table 1.3 gives clay's c_s as "< 100", prints moraine's as the constant
# 600, and leaves silt / sand / gravel as depth-dependent c_s(z̄); those four
# take a single value (see the module docstring).
# ``chalk`` and ``limestone`` are that table's shelf/slope rock rows: the
# porous, partly lithified calcareous rock of the chalk-to-limestone
# transition, not massive limestone. Hamilton (1980) Fig. 24 p. 1334, the
# velocity-density figure behind them, draws ONE calcareous trend and names
# four stretches of it — CALCAREOUS SEDIMENT (1.7-1.9 km/s), LIMESTONE AND
# CHALK (3.2-3.4), LIMESTONE (5.1-5.2), CALCITE (6.5, the pure mineral at
# 2.71 g/cm3). Table 1.3's chalk (2400 m/s) and limestone (3000) fall in the
# LIMESTONE AND CHALK stretch, Ainslie Table 4.20's limestone (5350 m/s at
# 2700 kg/m3) in the LIMESTONE one: the same word at different points along
# one porosity trend, which is why this row does not take Ainslie's the way
# ``granite`` does. What makes that a naming difference rather than a
# disagreement is his *chalk* row — identical to Table 1.3's in all five
# columns while his limestone sits 2350 m/s away; two compilations that agree
# to the digit on one rock are not measuring the next one differently.
# Hamilton's own in-situ marine limestone is the slow one: Table IV p. 1320
# fits the Ontong-Java "calcareous ooze, chalk, limestone" section as
# V = 1.559 + 1.713 Z - 0.374 Z**2 km/s (Z in km below the seabed), i.e.
# 2.90 km/s a full kilometre in, and it never reaches 5 km/s. For a massive
# limestone basement pass explicit values rather than this preset.
# ``granite`` has no row in Table 1.3. It is the granite row of Ainslie,
# *Principles of Sonar Performance Modelling* (Springer Praxis, 2010),
# Table 4.20 p. 183, "Representative geoacoustic parameters for typical
# sedimentary and igneous rocks": rho 2650 kg/m3, c_p 5750 m/s, alpha_p
# 0.10 dB/lambda, c_s 3000 m/s, alpha_s 0.20 dB/lambda — rounded there to the
# nearest 50 kg/m3, 50 m/s and 0.05 dB/lambda, with the text warning that "the
# variation around these representative values can be large". He names the
# table's own sources on that page: "The main sources used to construct Table
# 4.20 are Carmichael (1982) for wave speeds and Jensen et al. (1994) for
# attenuation", with Christensen and Salisbury (1975) for basalt, Assefa and
# Sothcott (1997) and Hamilton (1979) besides.
# Its rows and Table 1.3's are separate compilations that disagree where
# they overlap — Ainslie's limestone row is 2.70 / 5350 / 2400 against
# Table 1.3's 2.4 / 3000 / 1500 — so granite takes Ainslie's row entire, and
# no other row takes anything from it. Each row is the source the index above
# names for it, never a blend.
# What the wave speeds stand on. Carmichael, *Handbook of Physical Properties
# of Rocks* Vol. II, Table 18 p. 142, gives twelve water-saturated granite
# samples: c_p 5.10 to 6.30 km/s (mean 5.62) at densities 2.62 to 2.67 g/cm3.
# 5750 m/s at 2650 kg/m3 is one draw from that spread, which is Ainslie's
# "can be large" made quantitative: the whole 1.2 km/s of it sits inside
# 0.05 g/cm3 of density, so knowing a granite's density does not pin its
# sound speed to better than that. How much of c_p is pore state rather than
# mineral is visible in the same twelve rows: dry, they read 3.20 to
# 5.35 km/s, up to 1.9 km/s below their own saturated value.
# Two limits on granite's attenuations. They are a rock-*class* figure, not a
# granite measurement: the attenuations come, by Ainslie's own attribution
# above, from Jensen et al. (1994), which carries no granite row, and his
# sandstone, basalt, granite and limestone rows all share alpha_p = 0.10 while
# basalt, granite and limestone all share alpha_s = 0.20. And they are
# low-frequency: resonant-bar measurements of water-saturated granite give
# Q ~ 30 at 100 kHz (Coyner & Martin 1990, read through Olson, Lyons & Saebo,
# JASA 139(4), 1833-1847 (2016), §II.A, where it sets delta_p ~ 0.02 and
# delta_s = 2 delta_p), i.e. alpha_p ~ 0.55 and alpha_s ~ 1.09 dB/lambda —
# 5.5 times the tabulated pair. The "Generic Granite" column of that paper's
# Table II — Bourbie, Coussy & Zinszner (1987) Table 5.2, quoted there —
# gives 0.27 and 1.36 dB/lambda, 2.7 and 6.8 times it. Pass explicit
# attenuations for a granite in the sonar band.
# The phi column comes from Hamilton & Bachman (1982) Table I — the row whose
# Table II density and velocity ratio reproduce the JKPS row, within 2.4 %. Those
# are per-class measured sample means, not class centres, hence 8.80 / 5.40 /
# 3.34 rather than round numbers. Gravel and moraine are coarser than anything
# in Hamilton's continental-terrace suite, so gravel takes TR 9407 Table 2's
# "Sandy Gravel" row, Mz = -1.0 — the coarsest ϕ either model in
# :mod:`uacpy.core.sediment` is fitted at — and moraine has no phi at all.
# That the row's Mz is -1.0 does not rest on reading the scan: the report's
# Eqs. 2-3 evaluated there give nu = 1.3370 and rho = 2.492, which are the
# digits printed in that row (Mz = -1.5 would give 1.3686 / 2.5873).
MATERIALS: Dict[str, Dict] = {
    # Unconsolidated sediments (c_s as Table 1.3 gives it: clay from its
    # "< 100", moraine as printed, silt / sand / gravel one value for c_s(z̄))
    'clay':      _entry(sound_speed=1500.0, density=1.5, attenuation=0.2,
                        shear_speed=80.0, shear_attenuation=1.0,
                        porosity=70.0, grain_size_phi=8.80),
    'silt':      _entry(sound_speed=1575.0, density=1.7, attenuation=1.0,
                        shear_speed=80.0, shear_attenuation=1.5,
                        porosity=55.0, grain_size_phi=5.40),
    'sand':      _entry(sound_speed=1650.0, density=1.9, attenuation=0.8,
                        shear_speed=110.0, shear_attenuation=2.5,
                        porosity=45.0, grain_size_phi=3.34),
    'gravel':    _entry(sound_speed=1800.0, density=2.0, attenuation=0.6,
                        shear_speed=180.0, shear_attenuation=1.5,
                        porosity=35.0, grain_size_phi=-1.0),
    'moraine':   _entry(sound_speed=1950.0, density=2.1, attenuation=0.4,
                        shear_speed=600.0, shear_attenuation=1.0,
                        porosity=25.0),
    # Rocks (which limestone the row is: see the note above)
    'chalk':     _entry(sound_speed=2400.0, density=2.2, attenuation=0.2,
                        shear_speed=1000.0, shear_attenuation=0.5),
    'limestone': _entry(sound_speed=3000.0, density=2.4, attenuation=0.1,
                        shear_speed=1500.0, shear_attenuation=0.2),
    'basalt':    _entry(sound_speed=5250.0, density=2.7, attenuation=0.1,
                        shear_speed=2500.0, shear_attenuation=0.2),
    'granite':   _entry(sound_speed=5750.0, density=2.65, attenuation=0.1,
                        shear_speed=3000.0, shear_attenuation=0.2),
}


def list_materials() -> List[str]:
    """Sorted list of preset names available in :data:`MATERIALS`."""
    return sorted(MATERIALS)


def get_material(name: str) -> Dict:
    """Return a copy of the preset dict for ``name`` (case-insensitive).

    Raises :class:`ConfigurationError` listing the available names if
    ``name`` is not in the catalog.
    """
    key = name.strip().lower()
    if key not in MATERIALS:
        raise ConfigurationError(
            f"Unknown material preset {name!r}. "
            f"Available: {list_materials()}"
        )
    return dict(MATERIALS[key])


__all__ = ["MATERIALS", "list_materials", "get_material"]
