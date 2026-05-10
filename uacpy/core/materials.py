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
              Mean grain size on the Wentworth ϕ scale (used by
              Bellhop/Kraken's ``acoustic_type='grain-size'`` bottom);
              ``None`` for consolidated rocks where ϕ is not defined.
roughness     RMS interface roughness (m); 0 unless overridden.
============= ================================================

``c_s`` for unconsolidated sediments (silt/sand/gravel) actually grows
with depth below the seabed; the presets use a near-surface (1 m)
value. Pass an explicit ``shear_speed`` if you need a different depth.
"""

from __future__ import annotations

from typing import Dict, List, Optional


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


# Wentworth ϕ midpoints used below: clay ≥ 8, silt 4–8, fine sand 2–3,
# medium sand 1–2, coarse sand 0–1, gravel −2..−1.
MATERIALS: Dict[str, Dict] = {
    # Fluid sediments (c_s tabulated at z = 1 m below the seabed)
    'clay':      _entry(sound_speed=1500.0, density=1.5, attenuation=0.2,
                        shear_speed=80.0, shear_attenuation=1.0,
                        porosity=70.0, grain_size_phi=9.0),
    'silt':      _entry(sound_speed=1575.0, density=1.7, attenuation=1.0,
                        shear_speed=80.0, shear_attenuation=1.5,
                        porosity=55.0, grain_size_phi=6.0),
    'sand':      _entry(sound_speed=1650.0, density=1.9, attenuation=0.8,
                        shear_speed=110.0, shear_attenuation=2.5,
                        porosity=45.0, grain_size_phi=2.0),
    'gravel':    _entry(sound_speed=1800.0, density=2.0, attenuation=0.6,
                        shear_speed=180.0, shear_attenuation=1.5,
                        porosity=35.0, grain_size_phi=-1.5),
    'moraine':   _entry(sound_speed=1950.0, density=2.1, attenuation=0.4,
                        shear_speed=600.0, shear_attenuation=1.0,
                        porosity=25.0),
    # Rocks
    'chalk':     _entry(sound_speed=2400.0, density=2.2, attenuation=0.2,
                        shear_speed=1000.0, shear_attenuation=0.5),
    'limestone': _entry(sound_speed=3000.0, density=2.4, attenuation=0.1,
                        shear_speed=1500.0, shear_attenuation=0.2),
    'basalt':    _entry(sound_speed=5250.0, density=2.7, attenuation=0.1,
                        shear_speed=2500.0, shear_attenuation=0.2),
    'granite':   _entry(sound_speed=5500.0, density=2.7, attenuation=0.1,
                        shear_speed=3000.0, shear_attenuation=0.2),
}


def list_materials() -> List[str]:
    """Sorted list of preset names available in :data:`MATERIALS`."""
    return sorted(MATERIALS)


def get_material(name: str) -> Dict:
    """Return a copy of the preset dict for ``name`` (case-insensitive).

    Raises :class:`KeyError` listing the available names if ``name`` is
    not in the catalog.
    """
    key = name.strip().lower()
    if key not in MATERIALS:
        raise KeyError(
            f"Unknown material preset {name!r}. "
            f"Available: {list_materials()}"
        )
    return dict(MATERIALS[key])


__all__ = ["MATERIALS", "list_materials", "get_material"]
