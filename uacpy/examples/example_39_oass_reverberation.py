"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 39: OASS Reverberation from a Rough Seabed
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE: Compute the reverberant field scattered from a rough water/sediment
           interface, and show why it is not transmission loss.

FEATURES: ✓ OASS reverberation level vs range
          ✓ The two-binary chain (mean field → scattered field) run by uacpy
          ✓ kind='reverberation' — a different quantity from TL, same dB unit
          ✓ c_low as a physical knob, not a numerical one

WHY TWO BINARIES: OASS computes nothing on its own. It integrates the
scattered field over the mean-field boundary operators a producer run leaves
in a `.rhs` file (unit 45, written with option 's'). `run()` drives the
producer first, into the same work dir, and hands its `.rhs` to `oass2`.
Pinning `work_dir` with `cleanup=False` keeps both decks on disk, which is
what makes the chain debuggable.
"""

import sys
import os
from pathlib import Path
# Repo root, so ``import uacpy`` resolves from a source checkout.
sys.path.insert(0, str(Path(__file__).parents[2]))

OUTPUT_DIR = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
                  or Path(__file__).parent / 'output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import uacpy  # noqa: E402
from uacpy.models import RunMode  # noqa: E402


def build_environment():
    """A 100 m isovelocity duct over a rough sand half-space.

    The roughness on the bottom carrier is what the mean-field run needs in
    order to write a non-empty `.rhs`; the *spectrum* (correlation length and
    exponent) is OASS's own, because OASES reads CL and M from the scattering
    deck rather than adopting them from the producer (oass.tex:182-183).
    """
    return uacpy.Environment(
        bathymetry=100.0,
        ssp=uacpy.SoundSpeedProfile(depths=[0.0, 100.0],
                                    data=[1500.0, 1500.0]),
        bottom=uacpy.Bottom.from_halfspace(uacpy.BoundaryProperties(
            sound_speed=1700.0, density=1.8, attenuation=0.5,
            roughness=0.5)),
    )


def main():
    env = build_environment()
    source = uacpy.Source(depths=50.0, frequencies=250.0)
    receiver = uacpy.Receiver(depths=[10.0, 50.0, 90.0],
                              ranges=np.linspace(100.0, 5000.0, 40))

    work_dir = OUTPUT_DIR / 'example_39_work'

    # interface=3 is the water/sediment boundary: deck layer 1 is the vacuum
    # upper half-space, 2 the water column, 3 the sediment. uacpy refuses an
    # interface that is not a bottom layer, because attaching the spectrum to
    # the wrong record leaves a POSITIVE RG there — an infinite correlation
    # length, i.e. no back-scatter at all, from a run that still exits 0.
    model = uacpy.OASS(
        interface=3,
        correlation_length=10.0,
        spectral_exponent=2.0,
        rms_roughness=0.5,
        work_dir=work_dir,
        cleanup=False,
        verbose=True,
    )

    reverb = model.run(env, source, receiver,
                       run_mode=RunMode.REVERBERATION)

    print(f"\n  kind = {reverb.kind!r}   unit = {reverb.unit!r}")
    print(f"  grid = {reverb.data.shape}  (depth x range)")
    print("\n  Both decks are on disk for inspection:")
    for name in sorted(p.name for p in work_dir.iterdir()):
        print(f"    {name}")

    # The quantity is TAGGED, not derived. A reverberation level shares TL's
    # dB representation exactly, so nothing about the array distinguishes it —
    # which is why putting the two on one colour scale is refused rather than
    # silently allowed.
    fig, ax = plt.subplots(figsize=(10, 5))
    reverb.plot(ax=ax, env=env,
                title='OASS — reverberation level, 250 Hz over rough sand')
    out = OUTPUT_DIR / 'example_39_oass_reverberation.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  ✓ Saved: {out}")

    print("\n  Note: c_low is physical here, not a tuning knob. It defaults to")
    print("  the mean field's own value; raising it truncates the scattering")
    print("  integral (a measured case moved 30 dB), while lowering it is")
    print("  inert because REVINT/REVCOV bound their own buffer reads.")


if __name__ == '__main__':
    main()
