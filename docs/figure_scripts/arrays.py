"""Figures for ``docs/guide/arrays.md``.

The code here is the code shown on the page: each builder is the worked
example, so a figure cannot drift from the snippet that claims to produce it.

Every scenario is synthetic plane-wave data on a uniform line array, because
the point of the page is what the processor does with a *known* bearing. Every
figure that depends on noise or snapshot count states both, and every random
draw comes from a seeded ``default_rng``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from uacpy.acoustic_signal import (
    bartlett_spectrum,
    beamform,
    music_spectrum,
    mvdr_spectrum,
    sample_covariance,
    shading_taper,
    steering_vectors,
)

# Write into docs/guide/figures/ rather than docs/models/figures/.
GUIDE = True

# 1500 Hz in 1500 m/s water makes the wavelength exactly 1 m, so every spacing
# below reads directly as a fraction of a wavelength.
C = 1500.0
FREQ = 1500.0
LAMBDA = C / FREQ


# ── shared helpers ───────────────────────────────────────────────────────────


def line_array(n_elements: int, spacing: float) -> np.ndarray:
    """Element coordinates (m) of a uniform line array."""
    return np.arange(n_elements) * spacing


def norm_db(power) -> np.ndarray:
    """Power spectrum in dB, normalised to its peak, floored at -120 dB."""
    p = np.asarray(power, dtype=float)
    return 10.0 * np.log10(np.maximum(p, p.max() * 1e-12) / p.max())


def beampattern(positions, angles, *, steer_deg=0.0, weights=None):
    """Response in dB of a beamformer steered to ``steer_deg``.

    A single plane wave from ``steer_deg`` is a rank-one covariance
    ``R = a a^H``, so the conventional beamformer scanned across ``angles``
    traces out the beampattern itself.
    """
    a = steering_vectors(positions, [steer_deg], FREQ, C)[0]
    R = np.outer(a, a.conj())
    e = steering_vectors(positions, angles, FREQ, C)
    if weights is not None:
        e = e * weights
    return norm_db(bartlett_spectrum(R, e))


def width_3db(angles, pattern_db, *, steer_deg=0.0) -> float:
    """Full -3 dB width of the lobe centred on ``steer_deg``, in degrees.

    The peak is located by the steer direction rather than by ``argmax``: an
    array with grating lobes has several equal maxima.
    """
    peak = int(np.argmin(np.abs(angles - steer_deg)))
    below = pattern_db <= -3.0
    lo = angles[:peak][below[:peak]]
    hi = angles[peak:][below[peak:]]
    return float(hi[0] - lo[-1])


def peak_sidelobe_db(angles, pattern_db, *, steer_deg=0.0) -> float:
    """Highest sidelobe, in dB relative to the mainlobe peak."""
    peak = int(np.argmin(np.abs(angles - steer_deg)))
    right = peak
    while right + 1 < pattern_db.size and pattern_db[right + 1] <= pattern_db[right]:
        right += 1
    left = peak
    while left - 1 >= 0 and pattern_db[left - 1] <= pattern_db[left]:
        left -= 1
    return float(np.concatenate([pattern_db[:left], pattern_db[right:]]).max())


def plane_wave_snapshots(positions, angles_deg, *, snr_db, n_snapshots, rng,
                         powers_db=None):
    """Synthetic array data: independent plane waves in white element noise.

    Each source is scaled to **unit power per element**, so ``snr_db`` is the
    per-element signal-to-noise ratio of a 0 dB source before any array gain.
    ``powers_db`` offsets individual sources from that reference.
    """
    angles_deg = np.atleast_1d(np.asarray(angles_deg, dtype=float))
    if powers_db is None:
        powers_db = np.zeros(angles_deg.size)
    n = np.size(positions)
    x = np.zeros((n, n_snapshots), dtype=complex)
    for angle, level in zip(angles_deg, np.atleast_1d(powers_db)):
        # steering_vectors is unit-norm; sqrt(n) restores unit magnitude per
        # element so the power bookkeeping above holds.
        a = steering_vectors(positions, [angle], FREQ, C)[0] * np.sqrt(n)
        s = (rng.standard_normal(n_snapshots)
             + 1j * rng.standard_normal(n_snapshots)) / np.sqrt(2.0)
        x += 10.0 ** (level / 20.0) * np.outer(a, s)
    sigma = 10.0 ** (-snr_db / 20.0)
    x += sigma * (rng.standard_normal((n, n_snapshots))
                  + 1j * rng.standard_normal((n, n_snapshots))) / np.sqrt(2.0)
    return x


# ── the array manifold ───────────────────────────────────────────────────────


def steering_phase():
    """What a steering vector is, and where lambda/2 comes from.

    Left: the array samples a spatial sinusoid whose period along the array is
    ``lambda / sin(theta)``. Right: the phase step between neighbours, which is
    what has to stay inside a single turn.
    """
    positions = line_array(16, LAMBDA / 2)
    fine = np.linspace(positions[0], positions[-1], 2000)
    k = 2.0 * np.pi * FREQ / C
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.0, 4.0))

    for angle, colour in zip((20.0, 60.0, 90.0), ('C0', 'C1', 'C2')):
        u = np.sin(np.deg2rad(angle))
        e = steering_vectors(positions, [angle], FREQ, C)[0]
        ax_l.plot(fine, np.cos(k * fine * u), lw=1.0, alpha=0.45, color=colour)
        ax_l.plot(positions, np.real(e) * np.sqrt(positions.size), 'o',
                  ms=4.5, color=colour,
                  label=f'{angle:.0f}°  (λ/sin θ = {LAMBDA / u:.2f} m, '
                        f'{LAMBDA / u / (LAMBDA / 2):.1f} samples/period)')
    ax_l.set_xlabel('Element position along the array (m)')
    ax_l.set_ylabel('Re{steering vector} × √N')
    ax_l.set_title('The array samples the wavefront, d = λ/2 = 0.5 m',
                   fontweight='bold', fontsize=10)
    ax_l.set_ylim(-1.6, 1.9)
    ax_l.legend(title='Arrival angle', fontsize=7, title_fontsize=8,
                loc='upper right')
    ax_l.grid(True, alpha=0.3)

    angles = np.linspace(-90.0, 90.0, 721)
    for d_over_lambda, style in ((0.25, '-'), (0.5, '-'), (1.0, '-')):
        step = 2.0 * d_over_lambda * np.sin(np.deg2rad(angles))  # Δφ / π
        ax_r.plot(angles, step, style, lw=1.6,
                  label=f'd = {d_over_lambda:g}λ')
    ax_r.axhspan(-1.0, 1.0, color='0.85', zorder=0)
    ax_r.text(-88, 0.0, 'unambiguous\n|Δφ| ≤ π', fontsize=8, va='center')
    ax_r.set_xlabel('Arrival angle from broadside (deg)')
    ax_r.set_ylabel('Phase step between elements (× π rad)')
    ax_r.set_title('Δφ = 2π(d/λ)·sin θ', fontweight='bold', fontsize=10)
    ax_r.set_xlim(-90, 90)
    ax_r.set_xticks([-90, -45, 0, 45, 90])
    ax_r.legend(fontsize=8)
    ax_r.grid(True, alpha=0.3)

    fig.suptitle('The array manifold — a plane wave is a phase ramp',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


def grating_lobes():
    """Spatial aliasing: what element spacing does to the response.

    Top: three arrays of the *same* aperture, so the mainlobe is identical and
    the only difference is the extra lobes. Middle: the same patterns against
    u = sin θ, where they are periodic with period λ/d. Bottom: why the rule is
    λ/2 rather than λ — the aliases move when you steer.
    """
    angles = np.linspace(-90.0, 90.0, 3601)
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.5))

    # Nd is held at 8λ, which fixes the mainlobe width.
    spacings = [(0.5, 16), (1.0, 8), (2.0, 4)]
    for d_over_lambda, n in spacings:
        positions = line_array(n, d_over_lambda * LAMBDA)
        pattern = beampattern(positions, angles)
        axes[0].plot(angles, pattern, lw=1.4,
                     label=f'd = {d_over_lambda:g}λ, N = {n}  '
                           f'(−3 dB: {width_3db(angles, pattern):.1f}°)')
    axes[0].set_title('Same aperture (Nd = 8λ), three element spacings — '
                      'broadside', fontweight='bold', fontsize=10)
    axes[0].set_xlabel('Angle from broadside (deg)')

    # The manifold continued past |u| = 1. steering_vectors only spans the
    # visible region, by construction; the continuation is what makes the
    # periodicity visible.
    u = np.linspace(-3.0, 3.0, 6001)
    k = 2.0 * np.pi * FREQ / C
    for d_over_lambda, n in spacings:
        positions = line_array(n, d_over_lambda * LAMBDA)
        manifold = np.exp(1j * k * np.outer(u, positions)) / np.sqrt(n)
        a = manifold[np.argmin(np.abs(u))]
        axes[1].plot(u, norm_db(np.abs(manifold.conj() @ a) ** 2), lw=1.4,
                     label=f'd = {d_over_lambda:g}λ  '
                           f'(period λ/d = {1 / d_over_lambda:g})')
    axes[1].axvspan(-1.0, 1.0, color='0.88', zorder=0)
    axes[1].text(0.0, 2.0, 'visible region  |u| ≤ 1', ha='center',
                 va='bottom', fontsize=9)
    axes[1].set_title('The same patterns in u = sin θ, continued past the '
                      'visible region', fontweight='bold', fontsize=10)
    axes[1].set_xlabel('u = sin θ')
    axes[1].set_xlim(-3, 3)

    steer = 45.0
    for d_over_lambda in (0.5, 0.75, 1.0):
        positions = line_array(16, d_over_lambda * LAMBDA)
        pattern = beampattern(positions, angles, steer_deg=steer)
        alias = np.sin(np.deg2rad(steer)) - 1.0 / d_over_lambda
        tag = ('no alias in view' if abs(alias) > 1.0
               else f'alias at {np.rad2deg(np.arcsin(alias)):.0f}°')
        axes[2].plot(angles, pattern, lw=1.4,
                     label=f'd = {d_over_lambda:g}λ  ({tag})')
    axes[2].axvline(steer, color='k', ls='--', lw=0.9, alpha=0.6)
    axes[2].set_title('N = 16 steered to +45° — the aliases move with the beam',
                      fontweight='bold', fontsize=10)
    axes[2].set_xlabel('Angle from broadside (deg)')

    for ax in axes:
        ax.set_ylabel('Response (dB)')
        ax.set_ylim(-60, 3)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
    axes[1].set_ylim(-40, 9)
    for ax in (axes[0], axes[2]):
        ax.set_xlim(-90, 90)
        ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])

    fig.suptitle('Grating lobes — spatial aliasing on a line array',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


# ── conventional beamforming ─────────────────────────────────────────────────


def beampattern_anatomy():
    """The conventional beampattern, and how it scales with aperture."""
    angles = np.linspace(-90.0, 90.0, 7201)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.0, 4.2))

    positions = line_array(16, LAMBDA / 2)
    pattern = beampattern(positions, angles)
    width = width_3db(angles, pattern)
    psl = peak_sidelobe_db(angles, pattern)
    null = np.rad2deg(np.arcsin(LAMBDA / (16 * LAMBDA / 2)))

    ax_l.plot(angles, pattern, lw=1.5, color='C0')
    ax_l.axhline(-3.0, color='0.5', ls=':', lw=0.9)
    ax_l.annotate(f'−3 dB width {width:.2f}°',
                  xy=(width / 2, -3.0), xytext=(28.0, -8.0), fontsize=8,
                  arrowprops=dict(arrowstyle='->', lw=0.8))
    ax_l.annotate(f'first null {null:.2f}°',
                  xy=(null, -40.0), xytext=(30.0, -33.0), fontsize=8,
                  arrowprops=dict(arrowstyle='->', lw=0.8))
    ax_l.annotate(f'first sidelobe {psl:.1f} dB',
                  xy=(-10.6, psl), xytext=(-86.0, -8.0), fontsize=8,
                  arrowprops=dict(arrowstyle='->', lw=0.8))
    ax_l.set_title('N = 16, d = λ/2, steered broadside',
                   fontweight='bold', fontsize=10)

    for n in (8, 16, 32):
        positions = line_array(n, LAMBDA / 2)
        pattern = beampattern(positions, angles)
        ax_r.plot(angles, pattern, lw=1.4,
                  label=f'N = {n:2d}, Nd = {n / 2:.0f}λ  '
                        f'(−3 dB: {width_3db(angles, pattern):5.2f}°)')
    ax_r.set_title('Doubling the aperture halves the mainlobe',
                   fontweight='bold', fontsize=10)
    ax_r.legend(fontsize=8, loc='lower right')

    for ax in (ax_l, ax_r):
        ax.set_xlabel('Angle from broadside (deg)')
        ax.set_ylabel('Response (dB)')
        ax.set_xlim(-90, 90)
        ax.set_xticks([-90, -45, 0, 45, 90])
        ax.set_ylim(-45, 3)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Conventional (Bartlett) beampattern',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


def shading():
    """The taper trade: sidelobe level against mainlobe width."""
    n = 32
    positions = line_array(n, LAMBDA / 2)
    angles = np.linspace(-90.0, 90.0, 7201)
    tapers = [('Rectangular (no shading)', None),
              ('Hann', 'hann'),
              ('Chebyshev, 50 dB', ('chebwin', 50))]

    fig = plt.figure(figsize=(9.0, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.42,
                          wspace=0.28)
    ax_w = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_full = fig.add_subplot(gs[1, :])

    rows = []
    for (label, window), colour in zip(tapers, ('C0', 'C1', 'C2')):
        weights = None if window is None else shading_taper(n, window)
        pattern = beampattern(positions, angles, weights=weights)
        width = width_3db(angles, pattern)
        psl = peak_sidelobe_db(angles, pattern)
        w = np.ones(n) if weights is None else weights
        gain_loss = 10.0 * np.log10(w.sum() ** 2 / (n * (w ** 2).sum()))
        rows.append((label, width, psl, gain_loss))

        ax_w.plot(np.arange(n), w, 'o-', ms=3, lw=1.2, color=colour,
                  label=label)
        for ax in (ax_zoom, ax_full):
            ax.plot(angles, pattern, lw=1.4, color=colour,
                    label=f'{label}  ({width:.2f}°, {psl:.1f} dB)')

    ax_w.set_xlabel('Element index')
    ax_w.set_ylabel('Weight')
    ax_w.set_title('The tapers (unit RMS)', fontweight='bold', fontsize=10)
    ax_w.legend(fontsize=8)
    ax_w.grid(True, alpha=0.3)

    ax_zoom.set_xlim(-12, 12)
    ax_zoom.set_ylim(-45, 3)
    ax_zoom.set_title('Mainlobe', fontweight='bold', fontsize=10)
    ax_zoom.set_xlabel('Angle from broadside (deg)')
    ax_zoom.set_ylabel('Response (dB)')
    ax_zoom.grid(True, alpha=0.3)

    ax_full.set_xlim(-90, 90)
    ax_full.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    ax_full.set_ylim(-100, 5)
    ax_full.set_xlabel('Angle from broadside (deg)')
    ax_full.set_ylabel('Response (dB)')
    ax_full.set_title('Full pattern — legend gives (−3 dB width, peak sidelobe)',
                      fontweight='bold', fontsize=10)
    ax_full.legend(fontsize=8, loc='upper right')
    ax_full.grid(True, alpha=0.3)
    ax_full.text(
        -88, 1.0,
        '\n'.join(f'{label}: array gain {loss:+.2f} dB re uniform'
                  for label, _, _, loss in rows),
        fontsize=8, va='top',
        bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.9))

    fig.suptitle('Array shading, N = 32, d = λ/2', fontweight='bold',
                 fontsize=12)
    return fig


# ── adaptive and subspace processors ─────────────────────────────────────────


def resolution():
    """Bartlett vs MVDR vs MUSIC on two sources — the money comparison."""
    positions = line_array(16, LAMBDA / 2)
    angles = np.linspace(-30.0, 30.0, 2401)
    steering = steering_vectors(positions, angles, FREQ, C)
    rayleigh = np.rad2deg(np.arcsin(LAMBDA / (16 * LAMBDA / 2)))
    # Two equal uncorrelated sources hold two Bartlett maxima down to 0.83 of
    # the Rayleigh scale; below that the dip between them reaches zero.
    merge = 0.83 * rayleigh

    cases = [
        ((-6.0, 6.0), None, f'12° apart — 1.7× the {rayleigh:.1f}° Rayleigh '
                            'scale: everything works'),
        ((-2.0, 2.0), None, f'4° apart — below the {merge:.1f}° merge point: '
                            'Bartlett has one maximum'),
        ((-10.0, 10.0), (0.0, -12.0), 'Unequal sources, 0 and −12 dB — '
                                      'only MVDR reports the level'),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.0), sharex=True)
    for ax, (bearings, powers, title) in zip(axes, cases):
        rng = np.random.default_rng(0)
        x = plane_wave_snapshots(positions, bearings, snr_db=10.0,
                                 n_snapshots=200, rng=rng, powers_db=powers)
        R = sample_covariance(x)
        for label, spectrum in (
                ('Bartlett', bartlett_spectrum(R, steering)),
                ('MVDR', mvdr_spectrum(R, steering)),
                ('MUSIC (n_sources=2)', music_spectrum(R, steering, 2))):
            ax.plot(angles, norm_db(spectrum), lw=1.4, label=label)
        for bearing in bearings:
            ax.axvline(bearing, color='k', ls='--', lw=0.9, alpha=0.45)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_ylabel('Normalised power (dB)')
        ax.set_ylim(-35, 3)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
    axes[-1].set_xlabel('Angle from broadside (deg)')
    axes[-1].set_xlim(-30, 30)

    fig.suptitle('Two plane waves, 16 elements, 200 snapshots, '
                 '10 dB per-element SNR', fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


def snapshots():
    """What MVDR costs: a covariance good enough to invert."""
    positions = line_array(16, LAMBDA / 2)
    n = positions.size
    angles = np.linspace(-30.0, 30.0, 2401)
    steering = steering_vectors(positions, angles, FREQ, C)
    bearings = (-2.0, 2.0)

    fig = plt.figure(figsize=(9.0, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.42,
                          wspace=0.28)
    ax_spec = fig.add_subplot(gs[0, :])
    ax_err = fig.add_subplot(gs[1, 0])
    ax_load = fig.add_subplot(gs[1, 1])

    for k in (8, 16, 32, 256):
        rng = np.random.default_rng(0)
        R = sample_covariance(plane_wave_snapshots(
            positions, bearings, snr_db=10.0, n_snapshots=k, rng=rng))
        ax_spec.plot(angles, norm_db(mvdr_spectrum(R, steering)), lw=1.4,
                     label=f'K = {k:3d}  (K/N = {k / n:g})')
    for bearing in bearings:
        ax_spec.axvline(bearing, color='k', ls='--', lw=0.9, alpha=0.45)
    ax_spec.set_title('MVDR as the snapshot count falls — sources at ±2°, '
                      '10 dB per-element SNR', fontweight='bold', fontsize=10)
    ax_spec.set_xlabel('Angle from broadside (deg)')
    ax_spec.set_ylabel('Normalised power (dB)')
    ax_spec.set_xlim(-30, 30)
    ax_spec.set_ylim(-35, 3)
    ax_spec.legend(fontsize=8, loc='upper right')
    ax_spec.grid(True, alpha=0.3)

    # How far the estimate is from the covariance it is trying to estimate.
    a = [steering_vectors(positions, [b], FREQ, C)[0] * np.sqrt(n)
         for b in bearings]
    R_true = (np.outer(a[0], a[0].conj()) + np.outer(a[1], a[1].conj())
              + 10.0 ** (-10.0 / 10.0) * np.eye(n))
    counts = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024])
    errors = []
    for k in counts:
        trials = []
        for trial in range(40):
            rng = np.random.default_rng(100 + trial)
            R = sample_covariance(plane_wave_snapshots(
                positions, bearings, snr_db=10.0, n_snapshots=int(k), rng=rng))
            trials.append(np.linalg.norm(R - R_true) / np.linalg.norm(R_true))
        errors.append(np.mean(trials))
    errors = np.array(errors)
    ax_err.loglog(counts, errors, 'o-', ms=4, lw=1.4, label='measured')
    ax_err.loglog(counts, errors[3] * np.sqrt(counts[3] / counts), ':',
                  color='0.4', lw=1.2, label=r'$\propto K^{-1/2}$')
    ax_err.axvspan(counts[0], n, color='0.88', zorder=0)
    ax_err.text(4.5, errors.min() * 1.3, 'K < N\nsingular', fontsize=8)
    ax_err.axvline(2 * n, color='C3', ls='--', lw=1.0)
    ax_err.text(2 * n * 1.15, errors.max() * 0.8, 'K = 2N', fontsize=8,
                color='C3')
    ax_err.set_xlabel('Snapshots K')
    ax_err.set_ylabel(r'$\|\hat{R}-R\|_F\,/\,\|R\|_F$')
    ax_err.set_title('Covariance estimate, mean of 40 trials',
                     fontweight='bold', fontsize=10)
    ax_err.legend(fontsize=8)
    ax_err.grid(True, which='both', alpha=0.3)

    wide = np.linspace(-90.0, 90.0, 3601)
    wide_steering = steering_vectors(positions, wide, FREQ, C)
    for loading in (0.0, 1e-6, 1e-2, 1.0):
        rng = np.random.default_rng(0)
        R = sample_covariance(plane_wave_snapshots(
            positions, bearings, snr_db=10.0, n_snapshots=12, rng=rng))
        spectrum = norm_db(
            mvdr_spectrum(R, wide_steering, diagonal_loading=loading))
        ax_load.plot(wide, spectrum, lw=1.4, label=f'loading = {loading:g}')
    for bearing in bearings:
        ax_load.axvline(bearing, color='k', ls='--', lw=0.9, alpha=0.45)
    ax_load.set_title('Diagonal loading at K = 12 (< N)',
                      fontweight='bold', fontsize=10)
    ax_load.set_xlabel('Angle from broadside (deg)')
    ax_load.set_ylabel('Normalised power (dB)')
    ax_load.set_xlim(-90, 90)
    ax_load.set_xticks([-90, -45, 0, 45, 90])
    ax_load.set_ylim(-40, 3)
    ax_load.legend(fontsize=8, loc='lower right')
    ax_load.grid(True, alpha=0.3)

    fig.suptitle('MVDR needs a covariance it can invert',
                 fontweight='bold', fontsize=12)
    return fig


def music_order():
    """MUSIC needs the source count — and gets it wrong in one direction."""
    positions = line_array(16, LAMBDA / 2)
    angles = np.linspace(-60.0, 60.0, 4801)
    steering = steering_vectors(positions, angles, FREQ, C)
    bearings = (-3.0, 3.0)

    rng = np.random.default_rng(0)
    R = sample_covariance(plane_wave_snapshots(
        positions, bearings, snr_db=10.0, n_snapshots=200, rng=rng))
    eigenvalues = np.linalg.eigvalsh(R)[::-1]

    fig, (ax_e, ax_s) = plt.subplots(1, 2, figsize=(9.0, 4.0))
    levels = 10.0 * np.log10(eigenvalues / eigenvalues.max())
    index = np.arange(1, levels.size + 1)
    ax_e.stem(index[:2], levels[:2], linefmt='C3-', markerfmt='C3o',
              basefmt=' ', label='signal subspace')
    ax_e.stem(index[2:], levels[2:], linefmt='C0-', markerfmt='C0o',
              basefmt=' ', label='noise subspace')
    ax_e.set_xlabel('Eigenvalue index')
    ax_e.set_ylabel('Eigenvalue (dB re largest)')
    ax_e.set_title('Eigenvalues of R — the split is the source count',
                   fontweight='bold', fontsize=10)
    ax_e.legend(fontsize=8)
    ax_e.grid(True, alpha=0.3)

    for n_sources, style in ((1, '-'), (2, '-'), (4, (0, (1.5, 1.5)))):
        label = {1: '1 — too few', 2: '2 — correct', 4: '4 — too many'}[n_sources]
        ax_s.plot(angles, norm_db(music_spectrum(R, steering, n_sources)),
                  lw=1.6, ls=style, label=f'n_sources = {label}')
    ax_s.plot(angles, norm_db(bartlett_spectrum(R, steering)), lw=1.0,
              ls='--', color='0.45', label='Bartlett, for reference')
    for bearing in bearings:
        ax_s.axvline(bearing, color='k', ls='--', lw=0.9, alpha=0.45)
    ax_s.set_xlabel('Angle from broadside (deg)')
    ax_s.set_ylabel('Normalised pseudospectrum (dB)')
    ax_s.set_title('MUSIC with the wrong source count',
                   fontweight='bold', fontsize=10)
    ax_s.set_xlim(-40, 40)
    ax_s.set_ylim(-22, 3)
    ax_s.legend(fontsize=8, loc='lower right')
    ax_s.grid(True, alpha=0.3)

    fig.suptitle('MUSIC — two sources at ±3°, 16 elements, 200 snapshots, '
                 '10 dB per-element SNR', fontweight='bold', fontsize=11)
    fig.tight_layout()
    return fig


def bearing_time():
    """A bearing-time record: a moving target beside a loud interferer."""
    positions = line_array(16, LAMBDA / 2)
    scan = np.linspace(-60.0, 60.0, 481)
    steering = steering_vectors(positions, scan, FREQ, C)
    n_blocks, n_snapshots = 100, 64
    track = np.linspace(-40.0, 30.0, n_blocks)

    rng = np.random.default_rng(0)
    conventional = np.empty((scan.size, n_blocks))
    adaptive = np.empty((scan.size, n_blocks))
    for t, bearing in enumerate(track):
        block = plane_wave_snapshots(
            positions, [bearing, 8.0], snr_db=0.0, n_snapshots=n_snapshots,
            rng=rng, powers_db=[0.0, 20.0])
        # beamform returns dB per snapshot; averaging in power over the block
        # is exactly bartlett_spectrum of the block's covariance.
        snr = beamform(block, positions, FREQ, angles=scan, SL=0.0, NL=0.0).snr
        conventional[:, t] = 10.0 * np.log10(np.mean(10.0 ** (snr / 10.0), axis=1))
        adaptive[:, t] = 10.0 * np.log10(
            mvdr_spectrum(sample_covariance(block), steering))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.8), sharey=True,
                             layout='constrained')
    vmax = max(conventional.max(), adaptive.max())
    for ax, data, label in ((axes[0], conventional, 'Conventional (beamform)'),
                            (axes[1], adaptive, 'MVDR')):
        mesh = ax.pcolormesh(np.arange(n_blocks), scan, data, cmap='viridis',
                             vmin=vmax - 40.0, vmax=vmax, shading='auto')
        ax.plot(np.arange(n_blocks), track, 'w--', lw=0.9, alpha=0.65)
        ax.axhline(8.0, color='w', ls=':', lw=0.9, alpha=0.65)
        ax.set_xlabel('Block')
        ax.set_title(label, fontweight='bold', fontsize=10)
    axes[0].set_ylabel('Bearing from broadside (deg)')
    axes[0].set_ylim(-60, 60)
    fig.colorbar(mesh, ax=axes, label='Output level (dB re element noise)',
                 pad=0.02)
    fig.suptitle('Bearing-time record — 0 dB target (dashed) crossing a '
                 '+20 dB interferer at 8° (dotted)\n'
                 '16 elements, 64 snapshots per block',
                 fontweight='bold', fontsize=11)
    return fig


FIGURES = {
    'arrays_steering_phase': steering_phase,
    'arrays_grating_lobes': grating_lobes,
    'arrays_beampattern': beampattern_anatomy,
    'arrays_shading': shading,
    'arrays_resolution': resolution,
    'arrays_snapshots': snapshots,
    'arrays_music_order': music_order,
    'arrays_bearing_time': bearing_time,
}
