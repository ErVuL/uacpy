"""
EXAMPLE 24: Synthesize a time series from H(f)
==============================================

End-to-end frequency → time-domain workflow:

1. Run Bellhop in BROADBAND mode → :class:`TransferFunction` H(d, r, f).
2. Build a Gaussian-windowed sinusoid as the source waveform p_s(t).
3. Call ``H.synthesize_time_series(p_s, fs)`` → :class:`TimeSeriesField`.
4. Plot |H(f)| at one cell and the matching synthesized p(t).

ENVIRONMENT
    Pekeris waveguide, single (range, depth) receiver point.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

import uacpy
from uacpy.core.environment import BoundaryProperties
from uacpy.models import Bellhop, RunMode

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    bottom = BoundaryProperties(
        acoustic_type='half-space', sound_speed=1700.0,
        density=1.5, attenuation=0.5,
    )
    env = uacpy.Environment(
        name='Pekeris', depth=100.0, sound_speed=1500.0, bottom=bottom,
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
    envelope = np.exp(-((t_src - duration / 2) ** 2) / (2 * sigma ** 2))
    p_src = envelope * np.sin(2 * np.pi * f_center * t_src)

    # 3. Synthesize p(t) = IFFT(H · S)
    ts = H.synthesize_time_series(p_src, sample_rate=fs)
    print(f"TimeSeriesField shape: {ts.data.shape}, "
          f"dt={ts.metadata['dt']*1e3:.3f} ms, nt={ts.n_t}")

    # 4. Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    H_trace = H.data[0, 0]
    TL_dB = -20.0 * np.log10(np.maximum(np.abs(H_trace), 1e-12))
    axes[0].plot(H.frequencies, TL_dB, 'C0-', lw=1.2)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('TL(f)  (dB)')
    axes[0].set_title(
        f'Transmission loss at r={target_range_m/1000:.1f} km, '
        f'z={target_depth_m:.0f} m'
    )
    axes[0].grid(True, alpha=0.3)

    p_trace = ts.data[0, 0]
    axes[1].plot(ts.time * 1e3, p_trace, 'C1-', lw=1.0)
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('p(t)')
    axes[1].set_title('Synthesized time series')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / 'example_24_synthesize_time_series.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
