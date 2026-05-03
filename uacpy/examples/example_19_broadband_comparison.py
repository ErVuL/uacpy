"""
===============================================================================
BROADBAND COMPARISON: Time-Series and Transfer Functions Across Models
===============================================================================

OBJECTIVE:
    Demonstrate broadband / time-series capability across all models that
    support it:
    - Bellhop: ray-tracing arrivals → transfer function or delay-and-sum
    - RAM (mpiramS, ramsurf1.5, rams0.5): native broadband PE on each of
      the three RAM backends — same physics, three independent codes
    - SPARC: time-marched FFP, returns time-domain pressure
    - Scooter: multi-frequency FFP, returns transfer function
    - KrakenField: multi-frequency normal modes, returns transfer function

SCENARIO:
    Pekeris waveguide (isovelocity, 100 m depth, 100 Hz center frequency).
    Compare transfer functions and synthesize time-domain impulse responses.

NOTE ON PRE-ARRIVAL SIDELOBES:
    Bellhop, Scooter and KrakenField use coarse frequency sampling here
    (32 points across 50–200 Hz → df ≈ 4.8 Hz → 1/df ≈ 207 ms time-domain
    period). When the IFFT window is wider than 1/df, periodic replicas
    of the impulse response appear before the geometric arrival r/c₀ —
    that's the "little bit of signal" visible upstream of 3333 ms.
    `TransferFunction.to_time_trace` interpolates the spectrum onto a finer FFT
    grid (df_fft ≤ 1 Hz) to suppress them, but residual ripple from the
    coarse source data remains. The three RAM backends use much denser
    frequency sampling (Q=2, T=5 → df = 0.2 Hz, ~250 freqs across the
    band) so their pre-arrival traces are essentially clean.

FEATURES DEMONSTRATED:
    - RunMode.BROADBAND for H(f) transfer functions across all models
    - RunMode.TIME_SERIES for time-domain p(t):
        * SPARC: native time-marching (no source waveform required)
        * Bellhop: delay-and-sum with source_waveform + sample_rate
        * RAM/Scooter/KrakenField: BROADBAND + IFFT-convolve with source waveform
    - Transfer function (complex pressure vs. frequency) output
    - IFFT synthesis for time-domain impulse response
    - Delay-and-sum convolution with LFM chirp source
    - Comparison of time-domain results across models

===============================================================================
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

import uacpy
from uacpy.models import Bellhop, RAM, SPARC, Scooter, KrakenField
from uacpy.models.base import RunMode


def main():
    # =========================================================================
    # 1. ENVIRONMENT SETUP
    # =========================================================================
    print("=" * 70)
    print("Example 25: Broadband Model Comparison")
    print("=" * 70)

    env = uacpy.Environment(
        name='Pekeris Waveguide',
        depth=100,
        sound_speed=1500,
        ssp_type='isovelocity'
    )

    source = uacpy.Source(depth=36, frequency=100)

    # Single range for broadband comparison (5 km)
    receiver = uacpy.Receiver(
        depths=np.linspace(5, 95, 12),
        ranges=np.array([5000.0])
    )

    frequencies = np.linspace(50, 200, 16)

    # Target receiver for time-series extraction
    target_depth = 50.0  # m
    target_range = 5000.0  # m

    print(f"\nEnvironment: {env.name}")
    print(f"  Depth: {env.depth} m, Sound speed: 1500 m/s")
    print(f"  Source: {source.depth[0]} m depth, {source.frequency[0]} Hz center")
    print(f"  Receiver: {len(receiver.depths)} depths, range = {receiver.ranges[0]/1000:.0f} km")
    print(f"  Frequencies: {frequencies[0]:.0f} - {frequencies[-1]:.0f} Hz ({len(frequencies)} points)")
    print(f"  Time-series target: depth={target_depth} m, range={target_range/1000:.0f} km")

    results = {}

    # =========================================================================
    # 2. BELLHOP BROADBAND (arrivals → transfer function)
    # =========================================================================
    print("\n--- Bellhop Broadband ---")
    try:
        bellhop = Bellhop(verbose=False)
        result_bellhop = bellhop.run(
            env, source, receiver,
            run_mode=RunMode.BROADBAND,
            frequencies=frequencies,
        )
        print(f"  Output type: {result_bellhop.field_type}")
        print(f"  Shape: {result_bellhop.data.shape} (depth x range x freq)")
        print(f"  Frequencies: {result_bellhop.frequencies[0]:.1f} - "
              f"{result_bellhop.frequencies[-1]:.1f} Hz")
        results['Bellhop'] = result_bellhop
    except Exception as e:
        print(f"  SKIPPED: {e}")

    # =========================================================================
    # 3. RAM BROADBAND — all three backends on the same physics
    # =========================================================================
    # The dispatcher routes by env: Pekeris fluid + flat surface → mpiramS
    # (Fortran broadband). To exercise ramsurf1.5 and rams0.5 on identical
    # physics, we route with negligible env tweaks: a flat altimetry line
    # (z = 0) forces ramsurf, a tiny-shear sediment layer (cs = 1 m/s)
    # forces rams. Both perturb the field at the µ-scale; the comparison
    # is what each binary's broadband path produces on the same Pekeris.
    from uacpy.core.environment import (
        BoundaryProperties, LayeredBottom, SedimentLayer,
    )

    fluid_bottom = BoundaryProperties(
        acoustic_type='half-space', sound_speed=1700.0,
        density=1.7, attenuation=0.5,
    )
    env_mp = uacpy.Environment(
        name='Pekeris-fluid-flat', depth=env.depth,
        sound_speed=1500.0, ssp_type='isovelocity', bottom=fluid_bottom,
    )
    env_rs = uacpy.Environment(
        name='Pekeris-fluid-altimetry', depth=env.depth,
        sound_speed=1500.0, ssp_type='isovelocity', bottom=fluid_bottom,
        altimetry=[(0.0, 0.0), (8000.0, 0.0)],
    )
    elastic_tiny_shear = LayeredBottom(
        layers=[SedimentLayer(
            thickness=10.0, sound_speed=1700.0, density=1.7,
            attenuation=0.5, shear_speed=1.0, shear_attenuation=0.0,
        )],
        halfspace=BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1700.0, density=1.7, attenuation=0.5,
            shear_speed=1.0, shear_attenuation=0.0,
        ),
    )
    env_ra = uacpy.Environment(
        name='Pekeris-elastic-tiny-shear', depth=env.depth,
        sound_speed=1500.0, ssp_type='isovelocity',
        bottom=elastic_tiny_shear,
    )

    # Only the broadband window (Q, T) is supplied. (dr, dz) are picked
    # by the Lytaev Padé-error optimizer from env + centre frequency;
    # zmax is sized to the seafloor plus an absorbing-layer wavelength
    # buffer. All three backends share these defaults so the only
    # visible difference between traces is the underlying PE algorithm.
    common_numerics = dict(Q=2.0, T=2.0)
    ram_specs = [
        ('RAM (mpiramS)', env_mp, dict(**common_numerics)),
        ('RAM (ramsurf1.5)', env_rs, dict(**common_numerics)),
        ('RAM (rams0.5)', env_ra,
         dict(**common_numerics, rams_theta=45.0)),
    ]

    for label, env_ram, ram_kwargs in ram_specs:
        print(f"\n--- {label} Broadband ---")
        try:
            ram = RAM(verbose=False, **ram_kwargs)
            print(f"  backend = {ram.select_backend(env_ram)}")
            result_ram = ram.run(env_ram, source, receiver,
                                 run_mode=RunMode.BROADBAND)
            print(f"  Output shape: {result_ram.data.shape}")
            print(f"  Frequencies: {result_ram.frequencies[0]:.1f} - "
                  f"{result_ram.frequencies[-1]:.1f} Hz")
            results[label] = result_ram
        except Exception as e:
            print(f"  SKIPPED: {type(e).__name__}: {e}")

    # =========================================================================
    # 4. SCOOTER BROADBAND (multi-frequency FFP)
    # =========================================================================
    print("\n--- Scooter Broadband ---")
    try:
        scooter = Scooter(verbose=False)
        result_scooter = scooter.run(
            env, source, receiver,
            run_mode=RunMode.BROADBAND,
            frequencies=frequencies
        )
        print(f"  Output shape: {result_scooter.data.shape} (depth x range x freq)")
        print(f"  Frequencies: {result_scooter.frequencies[0]:.1f} - {result_scooter.frequencies[-1]:.1f} Hz")
        results['Scooter'] = result_scooter
    except Exception as e:
        print(f"  SKIPPED: {e}")

    # =========================================================================
    # 5. KRAKENFIELD BROADBAND (multi-frequency normal modes)
    # =========================================================================
    print("\n--- KrakenField Broadband ---")
    try:
        kf = KrakenField(verbose=False)
        result_kf = kf.run(
            env, source, receiver,
            run_mode=RunMode.BROADBAND,
            frequencies=frequencies
        )
        print(f"  Output shape: {result_kf.data.shape} (depth x range x freq)")
        print(f"  Frequencies: {result_kf.frequencies[0]:.1f} - {result_kf.frequencies[-1]:.1f} Hz")
        results['KrakenField'] = result_kf
    except Exception as e:
        print(f"  SKIPPED: {e}")

    # =========================================================================
    # 6. SPARC TIME-DOMAIN (direct time-marching)
    # =========================================================================
    print("\n--- SPARC Time-Domain ---")
    try:
        sparc = SPARC(verbose=False)
        # SPARC needs a horizontal array (single depth, multiple ranges)
        receiver_sparc = uacpy.Receiver(
            depths=np.array([50.0]),
            ranges=np.linspace(500, 5000, 5)
        )
        result_sparc = sparc.run(
            env, source, receiver_sparc, run_mode=RunMode.TIME_SERIES,
            f_min=50.0, f_max=200.0,
            n_t_out=1001,
            t_max=4.0,
        )
        print(f"  Output shape: {result_sparc.data.shape} (n_depths x nr x nt)")
        print(f"  Time step: {result_sparc.metadata['dt']*1000:.3f} ms")
        print(f"  Duration: {result_sparc.metadata['time'][-1]*1000:.1f} ms")
        results['SPARC'] = result_sparc
    except Exception as e:
        print(f"  SKIPPED: {e}")

    # =========================================================================
    # 7. BELLHOP DELAY-AND-SUM WITH LFM CHIRP
    # =========================================================================
    print("\n--- Bellhop Delay-and-Sum (LFM chirp) ---")
    try:
        bellhop_das = Bellhop(verbose=False)

        # Generate LFM chirp: 50-150 Hz over 100 ms
        fs = 2000.0  # sample rate
        chirp_duration = 0.1
        t_chirp = np.arange(0, chirp_duration, 1.0 / fs)
        f0, f1 = 50.0, 150.0
        chirp = np.sin(2 * np.pi * (f0 * t_chirp + (f1 - f0) / (2 * chirp_duration) * t_chirp**2))

        # Receiver with target depth/range for arrivals computation
        receiver_das = uacpy.Receiver(
            depths=np.array([target_depth]),
            ranges=np.array([target_range])
        )
        result_das = bellhop_das.run(
            env, source, receiver_das,
            run_mode=RunMode.TIME_SERIES,
            source_waveform=chirp,
            sample_rate=fs,
            depth=target_depth,
            range_m=target_range,
        )
        print(f"  Output type: {result_das.field_type}")
        print(f"  Shape: {result_das.data.shape}")
        print(f"  Time: {result_das.metadata['time'][0]*1000:.1f} - "
              f"{result_das.metadata['time'][-1]*1000:.1f} ms")
        print(f"  Max amplitude: {np.max(np.abs(result_das.data)):.6f}")
        results['Bellhop (chirp)'] = result_das
    except Exception as e:
        print(f"  SKIPPED: {e}")

    # =========================================================================
    # 8. COMPARISON PLOTS
    # =========================================================================
    if len(results) < 2:
        print("\nNot enough models succeeded for comparison. Exiting.")
        return

    # --- Plot A: Transfer function magnitude/phase comparison ---
    tf_models = {k: v for k, v in results.items()
                 if v.field_type == 'transfer_function'}
    if tf_models:
        from uacpy.visualization.plots import plot_transfer_function
        depth_idx = receiver.depths.shape[0] // 2
        depth_m = receiver.depths[depth_idx]
        fig, _ = plot_transfer_function(
            tf_models, depth_idx=depth_idx, range_idx=0,
            show_phase=True, unwrap_phase=False,
            title=f'Transfer functions — depth={depth_m:.0f} m, '
                  f'range={target_range/1000:.0f} km',
        )
        fig.savefig(OUTPUT_DIR / 'example_19_transfer_functions.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\nSaved: {OUTPUT_DIR / 'example_19_transfer_functions.png'}")

    # --- Plot B: TL vs depth comparison at center frequency ---
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, result in tf_models.items():
        freqs = result.frequencies
        center_idx = np.argmin(np.abs(freqs - source.frequency[0]))
        # New shape (n_d, n_r, n_f) → pick first range, freq slice on axis 2.
        tl_at_fc = -20 * np.log10(np.abs(result.data[:, 0, center_idx]) + 1e-30)
        ax.plot(tl_at_fc, result.depths, label=name, linewidth=1.5)

    ax.set_xlabel('Transmission Loss (dB)')
    ax.set_ylabel('Depth (m)')
    ax.set_title(f'TL vs Depth at {source.frequency[0]:.0f} Hz, '
                 f'range={target_range/1000:.0f} km')
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'example_19_tl_depth_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'example_19_tl_depth_comparison.png'}")

    # --- Plot C: TIME-SERIES COMPARISON (all models at same receiver) ---
    # Convert all transfer functions to time domain at the target point
    # Use consistent t_start for all models so they share the same time window
    c0 = 1500.0
    t_arrival = target_range / c0  # ~3.333 s for 5 km
    t_start_common = max(0.0, t_arrival - 0.5)  # 0.5 s before first arrival

    print(f"\n--- Converting to time-domain at depth={target_depth} m, "
          f"range={target_range/1000:.0f} km ---")
    print(f"  Common t_start = {t_start_common*1000:.1f} ms "
          f"(arrival at {t_arrival*1000:.1f} ms)")

    ts_results = {}  # name → (time_array_ms, data_array)

    for name, result in tf_models.items():
        try:
            ts = result.to_time_trace(
                depth=target_depth, range_m=target_range,
                t_start=t_start_common,
            )
            ts_results[name] = (ts.time * 1000, ts.data)
            print(f"  {name}: {len(ts.data)} samples, "
                  f"t=[{ts.time[0]*1000:.1f}, "
                  f"{ts.time[-1]*1000:.1f}] ms")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    # Add Bellhop delay-and-sum result
    if 'Bellhop (chirp)' in results:
        r = results['Bellhop (chirp)']
        ts_results['Bellhop (chirp)'] = (r.metadata['time'] * 1000, r.data)

    # Add SPARC (depth 0, last range). New shape: (n_d, n_r, n_t) — pick
    # depth 0, last range, all time samples.
    if 'SPARC' in results:
        r = results['SPARC']
        ts_results['SPARC'] = (r.metadata['time'] * 1000, r.data[0, -1, :])

    if ts_results:
        # Separate impulse-response models from chirp/SPARC for cleaner comparison
        ir_results = {k: v for k, v in ts_results.items()
                      if k not in ('Bellhop (chirp)', 'SPARC')}
        other_results = {k: v for k, v in ts_results.items()
                         if k in ('Bellhop (chirp)', 'SPARC')}

        n_ts = len(ts_results)
        fig, axes = plt.subplots(n_ts, 1, figsize=(14, 2.5 * n_ts), squeeze=False)
        axes = axes[:, 0]

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        # Compute common time window for impulse-response models
        t_center_ms = t_arrival * 1000
        t_window_ms = 500  # +/- 250 ms around arrival
        t_lo = t_center_ms - t_window_ms / 2
        t_hi = t_center_ms + t_window_ms / 2

        for idx, (name, (t_ms, data)) in enumerate(ts_results.items()):
            ax = axes[idx]
            color = colors[idx % len(colors)]

            ax.plot(t_ms, data, color=color, linewidth=0.8)
            ax.set_ylabel(name, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Use common time window for comparable models
            if name in ir_results or name == 'SPARC':
                ax.set_xlim(t_lo, t_hi)

            # Show max amplitude
            max_amp = np.max(np.abs(data))
            ax.text(0.98, 0.92, f'max = {max_amp:.2e}',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=8, color='gray',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

        axes[-1].set_xlabel('Time (ms)')
        fig.suptitle(f'Time-Series Comparison — depth={target_depth:.0f} m, '
                     f'range={target_range/1000:.0f} km\n'
                     f'fc={source.frequency[0]:.0f} Hz, '
                     f'Pekeris waveguide {env.depth:.0f} m',
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / 'example_19_time_series_comparison.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {OUTPUT_DIR / 'example_19_time_series_comparison.png'}")

    # --- Plot D: Bellhop delay-and-sum detail ---
    if 'Bellhop (chirp)' in results:
        r = results['Bellhop (chirp)']
        t_ms = r.metadata['time'] * 1000
        data = r.data

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

        # Source chirp
        t_chirp_ms = t_chirp * 1000
        ax1.plot(t_chirp_ms, chirp, 'k-', linewidth=0.8)
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Source: LFM chirp {f0:.0f}–{f1:.0f} Hz, '
                      f'{chirp_duration*1000:.0f} ms')
        ax1.set_xlabel('Time (ms)')
        ax1.grid(True, alpha=0.3)

        # Received waveform
        ax2.plot(t_ms, data, 'b-', linewidth=0.8)
        ax2.set_ylabel('Amplitude')
        ax2.set_title(f'Received: Bellhop delay-and-sum at '
                      f'depth={target_depth:.0f} m, '
                      f'range={target_range/1000:.0f} km')
        ax2.set_xlabel('Time (ms)')
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / 'example_19_bellhop_chirp.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {OUTPUT_DIR / 'example_19_bellhop_chirp.png'}")

    # --- Plot E: SPARC waterfall (if available) ---
    if 'SPARC' in results:
        result_sparc = results['SPARC']
        time_ms = result_sparc.metadata['time'] * 1000
        # New shape (n_d, n_r, n_t) — depth 0 → (n_r, n_t).
        pressure = result_sparc.data[0]

        fig, ax = plt.subplots(figsize=(12, 6))
        n_traces = min(pressure.shape[0], 10)   # n_r
        ranges_km = receiver_sparc.ranges[:n_traces] / 1000

        for i in range(n_traces):
            trace = pressure[i, :]              # all time samples at range i
            trace_norm = trace / (np.max(np.abs(trace)) + 1e-30) * 0.8
            ax.plot(time_ms, trace_norm + i, 'k-', linewidth=0.7)
            ax.text(time_ms[-1] * 1.01, i, f'{ranges_km[i]:.1f} km',
                    fontsize=8, va='center')

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Range (trace index)')
        ax.set_title('SPARC Time-Domain Waveforms')
        ax.set_xlim(time_ms[0], time_ms[-1])
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / 'example_19_sparc_time_series.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {OUTPUT_DIR / 'example_19_sparc_time_series.png'}")

    # =========================================================================
    # 9. SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("BROADBAND MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Type':<20} {'Shape':<25} {'Notes'}")
    print("-" * 85)
    for name, result in results.items():
        notes = ''
        if result.field_type == 'transfer_function':
            notes = f'{len(result.frequencies)} freqs'
        elif result.field_type == 'time_series':
            notes = f"dt={result.metadata['dt']*1000:.2f} ms"
        print(f"{name:<20} {result.field_type:<20} {str(result.data.shape):<25} {notes}")

    print(f"\nModels with TIME_SERIES support:")
    print(f"  Bellhop     - arrivals → H(f) via Fourier synthesis, or delay-and-sum")
    print(f"  RAM         - native broadband PE (mpiramS), returns transfer_function")
    print(f"  Scooter     - multi-freq FFP (native freq loop), returns transfer_function")
    print(f"  KrakenField - multi-freq normal modes (Python loop), returns transfer_function")
    print(f"  SPARC       - time-marched FFP (native time domain), returns time_series")
    print(f"  OASP        - OASES PE broadband, returns transfer_function")


if __name__ == '__main__':
    main()
