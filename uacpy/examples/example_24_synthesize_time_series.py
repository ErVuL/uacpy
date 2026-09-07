"""
EXAMPLE 24: Synthesize a time series from H(f)
==============================================

End-to-end frequency → time-domain workflow:

1. Run Bellhop in BROADBAND mode → :class:`Field` H(d, r, f).
2. Build a Gaussian-windowed sinusoid as the source waveform p_s(t).
3. Call ``H.synthesize_time_series(p_s, fs)`` → :class:`Field`.
4. Plot |H(f)| at one cell and the matching synthesized p(t).

ENVIRONMENT
    Pekeris waveguide, single (range, depth) receiver point.
"""

import sys
import os
from pathlib import Path
# Repo root, so ``import uacpy`` resolves from a source checkout.
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import uacpy  # noqa: E402
from uacpy.core.environment import BoundaryProperties  # noqa: E402
from uacpy.models import Bellhop, RunMode  # noqa: E402
from uacpy.acoustic_signal.waveforms import gaussian_pulse  # noqa: E402

OUTPUT_DIR = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
                  or Path(__file__).parent / 'output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("\n" + "═" * 80)
    print("EXAMPLE 24: Synthesize a time series from H(f)")
    print("═" * 80)

    bottom = BoundaryProperties(
        acoustic_type='half-space', sound_speed=1700.0,
        density=1.5, attenuation=0.5,
    )
    env = uacpy.Environment(
        name='Pekeris', bathymetry=100.0, ssp=1500.0, bottom=bottom,
    )
    f_center = 200.0
    source = uacpy.Source(depths=20.0, frequencies=f_center)

    target_range_m = 5000.0
    target_depth_m = 50.0
    receiver = uacpy.Receiver(
        depths=np.array([target_depth_m]),
        ranges=np.array([target_range_m]),
    )

    # 1. Broadband H(f)
    frequencies = np.linspace(50.0, 400.0, 256)
    bellhop = Bellhop(verbose=False)
    H = bellhop.run(
        env, source, receiver,
        run_mode=RunMode.BROADBAND, frequencies=frequencies,
    )
    print(f"H shape: {H.data.shape}, freq range: "
          f"{H.frequencies[0]:.0f}–{H.frequencies[-1]:.0f} Hz")

    # 2. Gaussian-windowed sinusoid (5-cycle pulse at f_center)
    fs = 4000.0
    n_cycles = 5
    duration = n_cycles / f_center
    t_src = np.arange(0, duration, 1.0 / fs)
    sigma = duration / 6
    # exp(-(t - T/2)^2 / (2 sigma^2)): gaussian_pulse's width is sigma*sqrt(2)
    envelope = gaussian_pulse(t_src, duration / 2, sigma * np.sqrt(2))
    p_src = envelope * np.sin(2 * np.pi * f_center * t_src)

    # 3. Synthesize p(t) = IFFT(H · S)
    ts = H.synthesize_time_series(p_src, sample_rate=fs)
    print(f"Field shape: {ts.data.shape}, "
          f"dt={ts.dt*1e3:.3f} ms, nt={ts.n_times}")

    # 4. Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    # Each cut plots itself: TL(f) with the loss axis downward, p(t) linear.
    H.at(depth=target_depth_m, range=target_range_m).plot(
        ax=axes[0], color='C0', lw=1.2,
        title=(f'Transmission loss at r={target_range_m/1000:.1f} km, '
               f'z={target_depth_m:.0f} m'))
    ts.at(depth=target_depth_m, range=target_range_m).plot(
        ax=axes[1], color='C1', lw=1.0, title='Synthesized time series')

    fig.tight_layout()
    out = OUTPUT_DIR / 'example_24_synthesize_time_series.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {out}")

    print("\n✓ Example 24 complete\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
