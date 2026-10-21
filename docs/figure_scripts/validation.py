"""The plots under the README's validation table, one per benchmark row.

Each figure reruns the case its row's tests measure, on the case builders and
reference tables of those tests (imported from them, so the picture and the
number cannot drift apart), and overlays every engine of the row on the
reference. ``docs/models/validation.md`` gives the numbers, bounds and null
controls. One engine keeps one colour in every plot, and a second setting of
the same engine has its own. The reference is solid and the engines dashed,
each engine's dashes shifted along the line from the last one's, so engines
that overlap show through each other's gaps.
"""

from __future__ import annotations

import warnings

import numpy as np

import uacpy
from uacpy import Bellhop, Kraken, RAM, Receiver, RunMode, Scooter, Source
from uacpy.acoustics import bottom_loss_curve
from uacpy.models import OASR, OAST, SPARC, Bounce
from uacpy.tests import test_benchmarks_analytic as analytic_cases
from uacpy.tests import test_benchmarks_published as published
from uacpy.tests import test_benchmarks_sparc as sparc_cases

# The README shows the plots two per row, about 400 px wide; this canvas keeps
# the axis and legend text readable at that scale and full size one click away,
# and 160 dpi (about 850 px) keeps it sharp on a screen of twice the pixel
# density.
THUMB = (5.3, 3.4)
FIGURE_DPI = 160
REFERENCE_STYLE = dict(color='black', linewidth=1.6)
ENGINE_COLOURS = {'Kraken': 'C0', 'Scooter': 'C1', 'OAST': 'C2', 'OASR': 'C2',
                  'RAM': 'C3', 'Bellhop': 'C4', 'SPARC': 'C5', 'Bounce': 'C6'}
SECOND_SETTING_COLOURS = {'Kraken coupled': 'C9', 'RAM (ramgeo)': 'C8'}
# matplotlib's own dash and gap (lines.dashed_pattern, in line widths). The
# n engines of a figure start their dashes 1/n of a period apart: where they
# share a curve, an engine drawn first shows in the gaps of those drawn over
# it, and an engine alone leaves the reference showing in its gaps.
DASH = 3.7
GAP = 1.6


def _style(label, turn, n_engines):
    """Line style of engine ``label``, the ``turn``-th of ``n_engines``
    drawn on one axes: its colour (a second setting of an engine,
    ``'Kraken coupled'`` or ``'RAM (ramgeo)'``, has its own) and dashes in
    their own place along the line."""
    colour = SECOND_SETTING_COLOURS.get(label, ENGINE_COLOURS[label.split()[0]])
    return dict(color=colour, linewidth=1.2,
                linestyle=(turn * (DASH + GAP) / n_engines, (DASH, GAP)))


def _table_field(table, row=0, *, depth):
    """Row ``row`` of a ``tests/data/benchmarks`` reference table as a TL
    Field over range."""
    ranges, tl = published._load_tl_table(table)
    return uacpy.Field(data=tl[row][np.newaxis, :],
                       coords={'depth': np.array([depth]), 'range': ranges},
                       kind='pressure', unit='dB')


def _tl_thumbnail(reference_label, reference, engines, title):
    """``reference`` in black under each ``(label, field)`` of ``engines``,
    TL against range, one ``uacpy.plot.compare`` call per line."""
    fig, ax = uacpy.plot.compare({reference_label: reference}, figsize=THUMB,
                                 title=title, **REFERENCE_STYLE)
    for turn, (label, field) in enumerate(engines):
        uacpy.plot.compare({label: field}, ax=ax, title=title,
                           **_style(label, turn, len(engines)))
    ax.legend(fontsize=7)
    return fig


def _run(model, env, source, receiver, **kwargs):
    """``model.run`` with its notices muted: the thumbnails show numbers,
    and each notice has its own test."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return model.run(env, source, receiver, **kwargs)


def pekeris():
    """Pekeris waveguide, 50 Hz: the trapped-mode sum and four engines."""
    env = analytic_cases._pekeris_env()
    source = Source(depths=analytic_cases._MODAL_ZS,
                    frequencies=analytic_cases.FREQ)
    receiver = Receiver(depths=[50.0], ranges=np.arange(2000.0, 10001.0, 20.0))
    engines = [(label, _run(model, env, source, receiver))
               for label, model in (('Kraken', Kraken()), ('Scooter', Scooter()),
                                    ('OAST', OAST()), ('RAM', RAM()))]
    return _tl_thumbnail('Modal sum', uacpy.analytic.pekeris(env, source, receiver),
                         engines, 'Pekeris waveguide, 50 Hz, 50 m')


def ideal_waveguide():
    """Pressure-release guide, 25 Hz: Kraken over a vacuum, RAM over a
    near-massless bottom, against the Dirichlet modal sum."""
    vacuum = analytic_cases._pressure_release_env(
        uacpy.BoundaryProperties(acoustic_type='vacuum'))
    soft = analytic_cases._pressure_release_env(uacpy.BoundaryProperties(
        acoustic_type='half-space', sound_speed=analytic_cases.C_W,
        density=analytic_cases.RHO_SOFT, attenuation=0.0))
    source, receiver = analytic_cases._pr_src_rcv()
    engines = [('Kraken', _run(Kraken(), vacuum, source, receiver)),
               ('RAM', _run(RAM(dr=5.0, dz=0.25, n_pade=6), soft, source,
                            receiver))]
    return _tl_thumbnail('Modal sum',
                         uacpy.analytic.ideal_waveguide(vacuum, source, receiver),
                         engines, 'Ideal waveguide, 25 Hz, 30 m')


def rigid_waveguide():
    """Rigid-bottom guide: SPARC's pulse response, Fourier-transformed and
    divided by the source spectrum at 37.5 Hz, against the modal sum."""
    env = uacpy.Environment(
        name='sparc-rigid-guide', bathymetry=sparc_cases._DEPTH,
        ssp=sparc_cases._C,
        bottom=uacpy.BoundaryProperties(acoustic_type='rigid'))
    f0 = sparc_cases._FC
    receiver = Receiver(depths=[35.0], ranges=np.arange(800.0, 1201.0, 5.0))
    trace = _run(SPARC(verbose=False, pulse_type='PN+N', output_mode='S',
                       n_time_samples=2048, freq_min=5.0, freq_max=75.0,
                       rmax_factor=7.0),
                 env, Source(depths=sparc_cases._ZS, frequencies=f0), receiver,
                 run_mode=RunMode.TIME_SERIES,
                 output_duration=sparc_cases._T_MAX)
    # The test's transform: SPARC writes its time axis in single precision,
    # which Field.to_transfer_function refuses as non-uniform.
    times = np.asarray(trace.coords['time'], dtype=float)
    k = np.argmin(np.abs(np.fft.rfftfreq(times.size, times[1] - times[0]) - f0))
    ratio = (np.fft.rfft(np.asarray(trace.data), axis=-1)[..., k]
             / np.fft.rfft(sparc_cases.sparc_pseudo_gaussian(times, f0))[k])
    sparc = trace.replace(data=ratio, coords={'depth': trace.coords['depth'],
                                              'range': trace.coords['range']},
                          kind='pressure', unit='Pa')
    reference = uacpy.analytic.ideal_waveguide(
        env, Source(depths=sparc_cases._ZS, frequencies=f0), receiver)
    return _tl_thumbnail('Modal sum', reference, [('SPARC', sparc)],
                         'Rigid-bottom waveguide, 37.5 Hz, 35 m')


def ideal_wedge():
    """ASA Problem I: Bellhop's line source against the Buckingham &
    Tolstoy series over the pressure-release wedge."""
    z_r = 30.0
    ranges = published.IDEAL_WEDGE_RANGES
    ranges = ranges[(200.0 - 0.05 * ranges) > z_r + 5.0]
    receiver = Receiver(depths=[z_r], ranges=ranges)
    bellhop = _run(Bellhop(n_beams=4000), published._ideal_wedge_env('vacuum'),
                   Source(depths=100.0, frequencies=25.0, source_type='line'),
                   receiver)
    series = uacpy.Field(
        data=published._ideal_wedge_line_tl(ranges, z_r, 'vacuum')[np.newaxis, :],
        coords={'depth': np.array([z_r]), 'range': ranges},
        kind='pressure', unit='dB')
    return _tl_thumbnail('Analytic series', series,
                         [('Bellhop (line source)', bellhop)],
                         'Ideal wedge, 25 Hz, 30 m')


def rayleigh():
    """Plane-wave reflection off the Pekeris seabed: Bounce and OASR
    against the Rayleigh coefficient."""
    env = analytic_cases._pekeris_env()
    source = Source(depths=20.0, frequencies=analytic_cases.FREQ)
    receiver = Receiver(depths=[50.0], ranges=[1000.0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bounce = Bounce(c_low=1400.0, c_high=20000.0).compute_reflection(
            env, source, receiver)
        oasr = OASR().run(env, source, receiver, run_mode=RunMode.REFLECTION)
    title = 'Rayleigh reflection, fluid half-space'
    fig, ax = bottom_loss_curve(
        env.bottom, water_sound_speed=analytic_cases.C_W,
        water_density=analytic_cases.RHO_WATER).plot(
            figsize=THUMB, label='Rayleigh', title=title, **REFERENCE_STYLE)
    for turn, (label, rc) in enumerate((('Bounce', bounce), ('OASR', oasr))):
        rc.plot(ax=ax, label=label, title=title, **_style(label, turn, 2))
    ax.legend(fontsize=7)
    return fig


def asa_wedge():
    """ASA upslope wedge, 25 Hz, at the 30 m receiver: RAM, Bellhop and
    Kraken's two mode sums against COUPLE07."""
    env = published._wedge_env()
    ranges, _ = published._load_tl_table(published.WEDGE_TABLE)
    source = Source(depths=published.WEDGE_ZS, frequencies=published.WEDGE_FREQ)
    receiver = Receiver(depths=[published.WEDGE_ZR[0]], ranges=ranges)
    engines = [(label, _run(model, env, source, receiver)) for label, model in (
        ('RAM', RAM(backend='mpirams', dr=5.0, dz=0.25, n_pade=6)),
        ('Bellhop', Bellhop()),
        ('Kraken adiabatic', Kraken(mode_coupling='adiabatic')),
        ('Kraken coupled', Kraken(mode_coupling='coupled')))]
    return _tl_thumbnail(
        'COUPLE07', _table_field(published.WEDGE_TABLE,
                                 depth=published.WEDGE_ZR[0]),
        engines, 'ASA upslope wedge, 25 Hz, 30 m')


def pe_workshop_3b():
    """NORDA PE Workshop 3b, 250 Hz, half a metre above a lossy bottom."""
    env = published._pe3b_env()
    ranges, _ = published._load_tl_table(published.PE3B_TABLE)
    source = Source(depths=published.PE3B_Z, frequencies=published.PE3B_FREQ)
    receiver = Receiver(depths=[published.PE3B_Z], ranges=ranges)
    engines = [(label, _run(model, env, source, receiver)) for label, model in (
        ('Kraken', Kraken()), ('RAM', RAM()), ('Scooter', Scooter()),
        ('Bellhop', Bellhop()))]
    return _tl_thumbnail(
        'COUPLE07', _table_field(published.PE3B_TABLE, depth=published.PE3B_Z),
        engines, 'NORDA PE Workshop 3b, 250 Hz, 99.5 m')


def corrugated_seafloor():
    """Square-wave corrugated seafloor, 25 Hz: one-way engines against
    COUPLE07's two-way solution."""
    env = published._corrugated_env()
    ranges, _ = published._load_tl_table(published.CORRUGATION_TABLE)
    source = Source(depths=18.0, frequencies=25.0)
    receiver = Receiver(depths=[50.0], ranges=ranges)
    engines = [(label, _run(model, env, source, receiver)) for label, model in (
        ('Kraken coupled', Kraken(mode_coupling='coupled')),
        ('RAM (mpirams)', RAM(backend='mpirams', dr=5.0, dz=0.25, n_pade=6)),
        ('RAM (ramgeo)', RAM(backend='ramgeo', dr=0.5, dz=0.25, n_pade=6)))]
    return _tl_thumbnail(
        'COUPLE07 (two-way)',
        _table_field(published.CORRUGATION_TABLE, depth=50.0),
        engines, 'Corrugated seafloor, 25 Hz, 50 m')


def bucker_waveguide():
    """Bucker's waveguide, 100 Hz: OAST, Kraken with leaky modes and RAM on
    a deep grid against the wavenumber-integration table."""
    env = published._bucker_env()
    ranges, _ = published._load_tl_table(published.BUCKER_TABLE)
    source = Source(depths=30.0, frequencies=100.0)
    receiver = Receiver(depths=[90.0], ranges=ranges)
    engines = [(label, _run(model, env, source, receiver)) for label, model in (
        ('OAST', OAST()),
        ('Kraken leaky_modes=True', Kraken(leaky_modes=True)),
        ('RAM zmax=2000', RAM(zmax=2000.0)))]
    return _tl_thumbnail(
        'Scooter table', _table_field(published.BUCKER_TABLE, depth=90.0),
        engines, 'Bucker waveguide, 100 Hz, 90 m')


def problem_xi():
    """Reverberation Modeling Workshop Problem XI at 3.5 kHz, where the
    water spans 233 wavelengths: Bellhop against the wavenumber integral."""
    frequency = 3500
    table = f'rmw_problem_xi_{frequency}hz_scooter.txt'
    ranges, _ = published._load_tl_table(table)
    bellhop = _run(Bellhop(), published._problem_xi_env(),
                   Source(depths=30.0, frequencies=float(frequency)),
                   Receiver(depths=[50.0], ranges=ranges))
    return _tl_thumbnail('Scooter table', _table_field(table, depth=50.0),
                         [('Bellhop', bellhop)],
                         'RMW Problem XI, 3.5 kHz, 50 m')


def pe_workshop_4c():
    """NORDA PE Workshop 4c, 25 Hz, at the 150 m receiver: up the slope and
    onto the shelf."""
    env = published._pe4c_env()
    ranges, _ = published._load_tl_table(published.PE4C_TABLE)
    source = Source(depths=600.0, frequencies=25.0)
    receiver = Receiver(depths=[150.0], ranges=ranges)
    engines = [(label, _run(model, env, source, receiver)) for label, model in (
        ('Kraken adiabatic', Kraken(mode_coupling='adiabatic')),
        ('RAM', RAM()),
        ('Kraken coupled', Kraken(mode_coupling='coupled')))]
    return _tl_thumbnail(
        'COUPLE07', _table_field(published.PE4C_TABLE, depth=150.0),
        engines, 'NORDA PE Workshop 4c, 25 Hz, 150 m')


FIGURES = {
    'validation_pekeris': pekeris,
    'validation_ideal_waveguide': ideal_waveguide,
    'validation_rigid_waveguide': rigid_waveguide,
    'validation_ideal_wedge': ideal_wedge,
    'validation_rayleigh': rayleigh,
    'validation_asa_wedge': asa_wedge,
    'validation_pe_workshop_3b': pe_workshop_3b,
    'validation_corrugated_seafloor': corrugated_seafloor,
    'validation_bucker_waveguide': bucker_waveguide,
    'validation_pe_workshop_4c': pe_workshop_4c,
    'validation_problem_xi': problem_xi,
}
