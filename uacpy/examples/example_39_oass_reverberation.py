"""OASS reverberation from a rough seabed.

The reverberant field scattered back from a rough water/sediment interface —
and why it is not transmission loss. OASS computes nothing on its own: it
integrates the scattered field over the mean-field boundary operators that a
producer run leaves in a .rhs file, so run() drives the producer first, into
the same work dir, and hands its .rhs to oass2. Pinning work_dir with
cleanup=False keeps both decks on disk, which is what makes the chain
debuggable.

Uses: uacpy.OASS(interface=, correlation_length=, spectral_exponent=,
rms_roughness=) · RunMode.REVERBERATION · Field.kind ('reverberation', a
different quantity from TL in the same dB unit) · result.plot()
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

# A 100 m isovelocity duct over a rough sand half-space. The roughness on the
# bottom carrier is what lets the mean-field run write a non-empty .rhs; the
# *spectrum* (correlation length and exponent) is OASS's own, because OASES
# reads CL and M from the scattering deck rather than from the producer
# (oass.tex:182-183).
env = uacpy.Environment(
    bathymetry=100.0,
    ssp=uacpy.SoundSpeedProfile(depths=[0.0, 100.0], data=[1500.0, 1500.0]),
    bottom=uacpy.Bottom.from_halfspace(uacpy.BoundaryProperties(
        sound_speed=1700.0, density=1.8, attenuation=0.5, roughness=0.5)),
)
source = uacpy.Source(depths=50.0, frequencies=250.0)
receiver = uacpy.Receiver(depths=[10.0, 50.0, 90.0],
                          ranges=np.linspace(100.0, 5000.0, 40))

work_dir = OUT / 'example_39_work'

# interface=3 is the water/sediment boundary: deck layer 1 is the vacuum upper
# half-space, 2 the water column, 3 the sediment. uacpy refuses an interface
# that is not a bottom layer, because attaching the spectrum to the wrong
# record leaves a POSITIVE RG there — an infinite correlation length, i.e. no
# back-scatter at all, from a run that still exits 0.
reverb = uacpy.OASS(interface=3, correlation_length=10.0,
                    spectral_exponent=2.0, rms_roughness=0.5,
                    work_dir=work_dir, cleanup=False).run(
    env, source, receiver, run_mode=uacpy.RunMode.REVERBERATION)

print(f"  kind={reverb.kind!r} unit={reverb.unit!r} grid={reverb.data.shape}")
print("  both decks left on disk: "
      + ", ".join(sorted(p.name for p in work_dir.iterdir())))

# The quantity is TAGGED, not derived. Reverberation is a LOSS, like TL: OASES
# writes -10·log10 E[|p_scat|²], so a larger number is a weaker scattered
# field. It shares TL's dB representation exactly, and nothing about the array
# distinguishes the two — which is why putting them on one colour scale is
# refused rather than silently allowed.
fig, ax = plt.subplots(figsize=(10, 5))
reverb.plot(ax=ax, env=env,
            title='OASS — reverberation loss, 250 Hz over rough sand')
fig.savefig(OUT / 'example_39_oass_reverberation.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

print("  c_low is physical here, not a tuning knob: it defaults to the mean "
      "field's own\n  value, raising it truncates the scattering integral (a "
      "measured case moved 30 dB),\n  and lowering it is inert because "
      "REVINT/REVCOV bound their own buffer reads.")
