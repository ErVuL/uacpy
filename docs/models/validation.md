# Validation — measured agreement with benchmarks

uacpy's engines are measured against references that do not come from the
engine itself: closed-form solutions, a two-way coupled-mode solution written
in the test suite, and tables computed by an independent code for the
standard range-dependent cases of the literature. This page lists every one of
those benchmarks: the case, its reference, what each engine measured against
it, the bound the test holds it to, and where an engine does not reach the
reference.

Each number on this page is the one its test states. The test is named next to
the number (`test_benchmarks_<name>.py::test_…`), and
`uacpy/tests/test_validation_page.py` fails when the named test, reference
table or section no longer states a quoted number, or when a benchmark test
has no entry here. *Measured* is the agreement the test recorded when its
bound was set; *bound* is what the test asserts. `|dTL|` is the absolute TL
difference from the reference over the test's cells, summarised by its
median, 90th percentile (p90) or maximum. A **null control** is the same
engine run with the feature under test removed (the slope, the step, the
lossy bottom); it must land outside the bound, or the bound would only show
that the engine produces a plausible waveguide field.

The models' own pages say when to reach for each one
([model index](README.md)); this page says how close each gets.

---

## 1. Closed forms

Module `uacpy/tests/test_benchmarks_analytic.py` and its siblings compare the
engines with analytic solutions evaluated in the test file.

### 1.1 Pekeris waveguide

The isovelocity waveguide over a fluid half-space, summed over its trapped
modes: Porter, *The KRAKEN Normal Mode Program* (SACLANTCEN SM-245, 1991),
eq. 2.19 (`test_benchmarks_analytic.py::test_kraken_tl_matches_pekeris_modal_sum`).
A 100 m guide at 50 Hz, absolute TL in dB re 1 m.

| Engine | Measured | Bound | Test |
|---|---|---|---|
| `Modes.modal_pressure_field` | absolute level, prefactor and density convention | median < 0.1 dB, max < 0.4 dB | `test_benchmarks_analytic.py::test_modal_pressure_field_matches_pekeris_modal_sum` |
| Kraken eigenvalues | `k_m` against the characteristic equation | `atol=1e-5` | `test_benchmarks_analytic.py::test_kraken_modes_match_pekeris_analytic` |
| Kraken TL | 0.02 dB median, 0.07 dB max | median < 0.1, max < 0.4 dB | `test_benchmarks_analytic.py::test_kraken_tl_matches_pekeris_modal_sum` |
| Scooter TL | 0.03 dB median, 0.13 dB p90 | median < 0.2, p90 < 0.6 dB | `test_benchmarks_analytic.py::test_scooter_tl_matches_pekeris_modal_sum` |
| OAST TL (strong-field cells) | 0.02 dB median | median < 0.3, p90 < 1.0 dB | `test_benchmarks_analytic.py::test_oast_tl_matches_pekeris_modal_sum` |
| OASP, broadband bin at 49.922 Hz | 0.0068 dB median, 0.023 dB p90 | median < 0.1, p90 < 0.3 dB | `test_benchmarks_oases.py::test_oasp_broadband_bin_matches_pekeris_modal_sum_at_the_bin_frequency` |
| RAM TL | 0.55 dB median | median < 1.5 dB | `test_benchmarks_analytic.py::test_ram_tl_matches_pekeris_modal_sum` |

RAM's level and field shape are pinned per backend in
`test_benchmarks_ram_level.py`, against the same modal sum:

| Check | Measured | Bound | Test |
|---|---|---|---|
| Shape (50 Hz, 100 m): `RAM − modal sum` about its own median | mpiramS 0.45 / 1.12 dB, ramgeo 0.47 / 1.15, ramsurf 0.47 / 1.15, rams 0.04 / 0.14 (median / p90) | median < 1.0, p90 < 2.0 dB | `test_benchmarks_ram_level.py::test_ram_field_shape_follows_the_modal_sum_up_to_a_common_offset` |
| Absolute level at 100 Hz in 100 m | mpiramS 0.26 / 0.76 dB, ramgeo 0.24 / 0.87, ramsurf 0.24 / 0.87 | median < 0.8, p90 < 1.5 dB | `test_benchmarks_ram_level.py::test_ram_absolute_level_matches_the_modal_sum_on_a_well_resolved_case` |
| rams over a 50 m/s shear seabed | 0.04 dB median, 0.14 dB p90, 0.53 dB max | median < 0.3, p90 < 0.6 dB | `test_benchmarks_ram_level.py::test_rams_absolute_level_matches_the_modal_sum_on_a_near_fluid_seabed` |

### 1.2 Ideal (Dirichlet) waveguide

Pressure-release surface and bottom: the modes are elementary sines and the
modal sum is closed-form.

| Engine | Measured | Bound | Test |
|---|---|---|---|
| Kraken over a `'vacuum'` bottom | 0.014 dB median, 0.091 dB max | median < 0.1, max < 0.4 dB | `test_benchmarks_analytic.py::test_kraken_vacuum_waveguide_matches_dirichlet_modal_sum` |
| RAM (`dr=5, dz=0.25, n_pade=6`) over a near-massless bottom | 0.012 dB median, 0.043 dB p90 | median < 0.15, p90 < 0.5 dB | `test_benchmarks_analytic.py::test_ram_pressure_release_waveguide_matches_dirichlet_modal_sum` |
| The reference at an exact mode cutoff | 1.4e-6 dB from a guide a hair shallower | < 1e-4 dB | `test_benchmarks_analytic.py::test_a_waveguide_exactly_at_a_cutoff_does_not_generate_the_cutoff_mode` |
| The exact (Hankel) modal sum against the far-field form | meet far from the source, part near it; an unknown form is refused | relative 1e-3 far, 1e-2 near | `test_benchmarks_analytic.py::TestTheExactModalSumIsTheAsymptoticOnesLimit::test_the_forms_meet_far_from_the_source`, `test_benchmarks_analytic.py::TestTheExactModalSumIsTheAsymptoticOnesLimit::test_the_forms_part_near_the_source`, `test_benchmarks_analytic.py::TestTheExactModalSumIsTheAsymptoticOnesLimit::test_an_unknown_form_is_refused` |

### 1.3 Rigid-bottom waveguide (SPARC)

SPARC runs only vacuum or rigid bottoms, so its reference is the isovelocity
guide with a pressure-release surface and a rigid bottom, whose modes are
also elementary (`test_benchmarks_sparc.py`).

| Check | Measured | Bound | Test |
|---|---|---|---|
| Snapshot field, Fourier-transformed and divided by the source spectrum, against the modal sum (30 cells) | common gain +0.02 dB; 0.17 dB median, 0.33 dB p90, 0.61 dB max | gain within 0.5 dB; median < 0.5, p90 < 0.8 dB | `test_benchmarks_sparc.py::test_sparc_deconvolved_spectrum_matches_the_rigid_guide_modal_sum` |
| `p(t)` against the image sum of a unit point source, output modes R, D, S | gain 0.9946 (-0.05 dB), correlation 0.9981 | gain within 0.3 dB, correlation > 0.99 | `test_benchmarks_sparc.py::test_sparc_p_of_t_is_the_unit_source_image_sum` |

### 1.4 Ideal wedge (ASA benchmark Problem I)

Buckingham & Tolstoy, JASA 87, 1511-1513 (1990): a 2.86° up-slope wedge with
pressure-release surface and bottom, 25 Hz. A rigid-bottom series is written
in the test as well. A one-way PE cannot be scored here: at a receiver
between the source and the apex the exact field is a standing wave, and RAM
reproduces only its outgoing half (5.03 dB from the exact solution, 0.94 dB
from the one-way series; `test_benchmarks_analytic.py::test_bellhop_ideal_wedge_matches_analytic`).

| Engine | Measured | Bound | Test |
|---|---|---|---|
| Bellhop, 6 receivers | shape to about 0.4 dB | median < 1.5, max < 2.5 dB, std < 0.7 dB | `test_benchmarks_analytic.py::test_bellhop_ideal_wedge_matches_analytic` |
| Bellhop, 4000 beams, pressure-release bottom | 0.63 / 1.72 dB at 30 m, 1.18 / 2.67 dB at 150 m (median / p90) | (1.0, 2.5) and (1.7, 3.5) | `test_benchmarks_published.py::test_bellhop_follows_the_ideal_wedge_at_both_published_receivers` |
| Bellhop, 4000 beams, rigid bottom | 0.72 / 1.82 dB at 30 m, 1.85 / 4.94 dB at 150 m | (1.1, 2.6) and (2.5, 6.0) | `test_benchmarks_published.py::test_bellhop_follows_the_ideal_wedge_at_both_published_receivers` |

Over a rigid bottom Bellhop sits 0.72 dB from the rigid series and 4.10 dB
from the pressure-release one, and the two series are 3.85 dB apart, so the
rigid series is checked by an independent method. Null control: the flat
200 m guide, 5.2-7.5 dB median from the wedge
(`test_benchmarks_published.py::test_bellhop_follows_the_ideal_wedge_at_both_published_receivers`).

### 1.5 Lloyd mirror, phase and source conventions

A deep, impedance-matched bottom leaves the direct path and its
pressure-release surface image.

| Check | Measured | Bound | Test |
|---|---|---|---|
| Bellhop coherent TL, hat beams | two-path sum | max < 0.05 dB (Gaussian beams max < 3.0 dB) | `test_benchmarks_analytic.py::test_bellhop_lloyd_mirror` |
| Bellhop SEMICOHERENT_TL is the Lloyd-shaded incoherent sum | coherent floor 0.05 dB | max < 0.5 dB | `test_benchmarks_analytic.py::test_semicoherent_is_the_lloyd_shaded_incoherent_sum` |
| Complex pressure phase, Bellhop and Scooter | sign convention of the `.shd` field | mean phase within 20°, mean magnitude ratio within 0.1 of 1 | `test_benchmarks_analytic.py::test_complex_field_phase_matches_lloyd_mirror` |
| Line source carries the 2-D π/4 | point and line sources land on the same residual | 10° | `test_benchmarks_analytic.py::test_bellhop_line_source_carries_the_2d_quarter_wave_phase` |
| Line source level at 1 m | mean level against `√(8πk)·(i/4)[H0(kR1) − H0(kR2)]` | within 1.0 dB | `test_benchmarks_analytic.py::test_bellhop_line_source_level_is_unit_amplitude_at_one_metre` |
| `source_type='scaled'` removes `10·log10(r)` exactly | through the model | `atol=0.01` dB | `test_benchmarks_analytic.py::test_scaled_cylindrical_removes_exactly_the_spreading_term` |

### 1.6 Plane-wave reflection

The Rayleigh coefficient of a fluid half-space (Brekhovskikh & Lysanov).

| Engine | Measured | Bound | Test |
|---|---|---|---|
| Bounce reflection magnitude, critical angle and plateau | residual 3.3e-16 | `atol=1e-9`; critical angle within 3° | `test_benchmarks_analytic.py::test_bounce_reflection_matches_rayleigh` |
| OASR reflection magnitude | about 1e-6 | `atol=1e-3` | `test_benchmarks_analytic.py::test_oasr_reflection_matches_rayleigh` |
| Bounce reflects the seabed, not the water above it | water depth leaves `R` unchanged | 0.02 in magnitude, 5° in phase | `test_benchmarks_analytic.py::TestBounceReflectsTheSeabedNotTheOcean::test_reflection_coefficient_is_independent_of_water_depth` |
| Bounce at a water speed other than 1500 m/s | Rayleigh | median magnitude error < 0.03, phase < 10° | `test_benchmarks_analytic.py::TestBounceReflectsTheSeabedNotTheOcean::test_matches_rayleigh_at_a_water_speed_other_than_1500` |

### 1.7 Ocean-physics formulas

No binary needed (`test_benchmarks_physics.py`).

| Check | Reference | Bound | Test |
|---|---|---|---|
| `sound_speed_mackenzie` | Mackenzie, JASA 70, 807-812 (1981): 1550.744 m/s at 25 °C, 35 ppt, 1000 m | 1e-3 m/s | `test_benchmarks_physics.py::test_mackenzie_published_check_value` |
| `sound_speed_mackenzie` over a (T, S, D) table | the nine-term equation | 1e-6 m/s | `test_benchmarks_physics.py::test_sound_speed_mackenzie_matches_the_equation_table` |
| `sound_speed_unesco` and `sound_speed_delgrosso` at 15 °C, 35 PSU, surface | 1506.675 and 1506.678 m/s | 1e-3 m/s | `test_benchmarks_physics.py::test_unesco_and_delgrosso_documented_pair_at_15c_35psu_surface` |
| `ConstantAbsorption` in Kraken and Scooter | loss of `(r/λ)·value` | 0.9-1.4 × the prediction | `test_benchmarks_physics.py::test_constant_absorption_adds_the_expected_loss` |

### 1.8 OASES products

`test_benchmarks_oases.py` checks the OASES programs other than OAST against
closed forms: the half-space surface-noise field (Cron & Sherman, JASA 34,
1962) and the first-order perturbation scattering of a pressure-release rough
interface (Thorsos & Jackson, JASA 86, 1989).

| Check | Measured | Bound | Test |
|---|---|---|---|
| OASN level in an infinitely deep ocean is `SSLEV` | −0.019 .. −0.021 dB over 81 sensors | within 0.2 dB | `test_benchmarks_oases.py::test_oasn_surface_noise_level_in_an_infinitely_deep_ocean_is_the_source_level` |
| OASN vertical coherence against the dipole-sheet closed form | difference 0.0042 median, 0.0065 p90, 0.0079 max | median < 0.02, max < 0.05 | `test_benchmarks_oases.py::test_oasn_vertical_coherence_matches_the_dipole_sheet_closed_form` |
| OASS reverberation loss against the perturbation integral, to 1.5 km | 0.063 dB median, 0.136 dB max | median < 0.25, max < 0.5 dB | `test_benchmarks_oases.py::test_oass_reverberation_loss_matches_the_first_order_perturbation_integral` |
| OASS: doubling the rms roughness | −6.02 dB at all 78 cells | within 0.05 dB of `20·log10(2)` | `test_benchmarks_oases.py::test_oass_scattered_intensity_scales_with_the_square_of_the_rms_roughness` |
| OASSP: doubling the rms roughness of one realisation | ratio 2.000000 | 1e-3 on the ratio | `test_benchmarks_oases.py::test_oassp_realisation_is_linear_in_the_rms_roughness` |
| OASSP ensemble range law | slope −41.9 dB/decade against −43.0; level −5.3 dB from the integral | slope within 3 dB/decade; level −5.3 ± 3.0 dB (a regression pin, not a physics claim) | `test_benchmarks_oases.py::test_oassp_ensemble_intensity_follows_the_perturbation_integral_range_law` |

---

## 2. Stepped waveguide: two-way coupled modes

`test_benchmarks_coupled_mode.py` solves a single depth step in a Dirichlet
waveguide by two-way mode matching (Jensen, Kuperman, Porter & Schmidt,
*Computational Ocean Acoustics*, Sect. 5.11.1, after Evans, JASA 74, 1983),
200 m deepening to 204 m, 25 Hz, receivers 30 and 77 m beyond the step.

The reference is checked before any engine is measured against it:

| Property | Measured | Bound | Test |
|---|---|---|---|
| No step: the flat modal sum | 5.7e-13 dB | < 1e-10 dB | `test_benchmarks_coupled_mode.py::test_a_stepless_coupled_mode_solution_is_the_flat_dirichlet_modal_sum` |
| Overlap integrals against quadrature | 1e-12 | < 1e-9 | `test_benchmarks_coupled_mode.py::test_the_mode_matching_overlap_integrals_match_numerical_quadrature` |
| Step face is pressure-release | face/overlap pressure falls 1.64e-2 → 2.38e-3 with mode reach | converging | `test_benchmarks_coupled_mode.py::test_the_matched_field_leaves_the_step_face_pressure_release` |
| Power flux conserved | residual of `1 − R − T` ≤ 6.7e-16; `S` unitary to 4.4e-15 | < 1e-12, < 1e-11 | `test_benchmarks_coupled_mode.py::test_step_scattering_conserves_modal_power_flux` |
| Reciprocity | `S` symmetric to 4.3e-15; exchange across the step 3.5e-15 | < 1e-11 | `test_benchmarks_coupled_mode.py::test_step_scattering_is_reciprocal` |
| Truncation converged | 1.3e-3 dB from the deepest truncation | < 1e-2 dB | `test_benchmarks_coupled_mode.py::test_the_reference_field_has_stopped_moving_at_the_retained_mode_count` |
| Axis refocusing against exact Hankel functions | 0.0088 dB median at D2 = 178 m (1.0127 dB without it) | < 0.05 dB | `test_benchmarks_coupled_mode.py::test_the_reference_refocuses_the_step_reflection_at_the_range_axis` |
| The 2 % step is weakly coupled | reflected power 2.27e-4; the 11 % step 0.1246 | < 1e-3 and > 0.05 | `test_benchmarks_coupled_mode.py::test_a_two_percent_step_is_weakly_coupled_and_a_ten_percent_step_is_not` |
| The step is visible | 2.69 dB median, 10.16 dB p90 from the stepless field | > 1.5 and > 5.0 dB | `test_benchmarks_coupled_mode.py::test_the_step_moves_the_reference_field_far_more_than_any_engine_bound` |

| Engine | Measured | Bound | Null control | Test |
|---|---|---|---|---|
| RAM mpiramS | 0.069 dB median, 0.197 dB p90 | median < 0.35, p90 < 0.9 dB | 2.702 / 10.135 dB | `test_benchmarks_coupled_mode.py::test_ram_stepped_waveguide_matches_coupled_mode_reference` |
| RAM ramgeo | 0.078 dB median, 0.255 dB p90 | median < 0.4, p90 < 1.0 dB | 2.696 / 10.141 dB | `test_benchmarks_coupled_mode.py::test_ramgeo_stepped_waveguide_matches_coupled_mode_reference` |
| Bellhop, vacuum bottom | 1.233 dB median, 3.098 dB p90 | median < 1.45, p90 < 3.9 dB | 2.607-2.773 / 9.981-10.122 dB | `test_benchmarks_coupled_mode.py::test_bellhop_stepped_waveguide_matches_coupled_mode_reference` |

Where the step reflects, a one-way PE loses the field: at the 11 % up-step
RAM sits 1.131 dB median / 2.986 dB p90 from the two-way reference on the
same grid that gave 0.069 dB at the 2 % step
(`test_benchmarks_coupled_mode.py::test_a_one_way_pe_loses_the_stepped_field_once_the_step_reflects`).

---

## 3. Published range-dependent benchmarks

`test_benchmarks_published.py` scores the engines on the cases the field uses
to judge a range-dependent model. The references were computed by codes uacpy
does not wrap and ship as tables under `uacpy/tests/data/benchmarks/`, each
with a header naming the problem, the run settings and the source. COUPLE07
(R. B. Evans, December 2007, OALIB `Modes/couple/couple07.zip`) is the
stepwise coupled-mode code behind the ASA benchmark curves (Jensen & Ferla,
JASA 87, 1499-1510, 1990); it carries no licence, so only its generated
tables are shipped. Each table is checked on its own before an engine meets it
(the `test_the_…_reference_table_…` tests below).

### 3.1 ASA upslope wedge, 25 Hz

Isovelocity water 200 m deep shoaling to the apex at 4 km over a 1700 m/s,
density 1.5 bottom with 0.5 dB/wavelength; source 100 m, receivers 30 and
150 m. Reference: COUPLE07 Test Case 2, `benchmarks/asa_wedge_25hz_couple07.txt`;
the same wedge is Fig. 6.8 of *Computational Ocean Acoustics*. The table's
two deep nulls at 30 m sit at 1.16 and 1.87 km
(`test_benchmarks_published.py::test_the_asa_wedge_reference_table_places_the_published_nulls`).

| Engine | RMS 30 m / 150 m | Bound | Null control | Test |
|---|---|---|---|---|
| RAM mpirams (dr 5 m, dz 0.25 m, Padé 6) | 0.230 / 0.213 dB, nulls on the reference sample | 0.5 dB | 6.16 / 5.83 dB | `test_benchmarks_published.py::test_ram_matches_the_coupled_mode_reference_in_the_asa_wedge` |
| RAM ramgeo (same grid) | 0.225 / 0.215 dB | 0.5 dB | 6.16 / 5.83 dB | `test_benchmarks_published.py::test_ram_matches_the_coupled_mode_reference_in_the_asa_wedge` |
| Kraken coupled (15 profiles) | 2.747 / 2.422 dB, nulls 1.18 / 1.87 km | 3.5 dB | about 5.8 dB | `test_benchmarks_published.py::test_kraken_mode_sums_follow_the_asa_wedge_within_their_method_error` |
| Kraken adiabatic | 2.215 / 2.517 dB, nulls 1.16 / 1.89 km | 3.3 dB | about 5.8 dB | `test_benchmarks_published.py::test_kraken_mode_sums_follow_the_asa_wedge_within_their_method_error` |
| Bellhop, default fan | 2.023 / 2.322 dB, nulls 1.14 / 1.86 km plus a spurious one at 1.56 km | 2.8 dB | 5.65 dB | `test_benchmarks_published.py::test_bellhop_follows_the_asa_wedge_within_a_ray_models_error_at_25_hz` |

RAM on its automatic grid lands at 0.37 / 0.66 dB in 44 s instead of 1.5 s,
the excess inside 500 m of the source, so the test pins the grid
(`test_benchmarks_published.py::test_ram_matches_the_coupled_mode_reference_in_the_asa_wedge`).

### 3.2 NORDA PE Workshop I, Test Case 3b, 250 Hz

A range-independent guide, 100 m of water over a 1590 m/s, density 1.2 bottom;
source and receiver both 99.5 m deep, 0.5 m above the seafloor. Reference:
COUPLE07 Test Case 1, `benchmarks/pe_workshop_3b_250hz_couple07.txt`
(Davis, White & Cavanagh, NORDA Technical Note 143, 1982). The table holds
seven deep nulls over a 63-113 dB field
(`test_benchmarks_published.py::test_the_pe_workshop_3b_reference_table_places_its_nulls`).

| Engine (defaults) | Median / p90 TL difference | Bound | Test |
|---|---|---|---|
| RAM | 0.211 / 1.09 dB | 0.4 / 1.6 dB | `test_benchmarks_published.py::test_wave_engines_match_the_pe_workshop_3b_reference` |
| Kraken | 0.273 / 1.07 dB | 0.45 / 1.6 dB | `test_benchmarks_published.py::test_wave_engines_match_the_pe_workshop_3b_reference` |
| Scooter | 0.459 / 2.54 dB | 0.7 / 3.3 dB | `test_benchmarks_published.py::test_wave_engines_match_the_pe_workshop_3b_reference` |
| Bellhop | 5.37 dB median, +4.08 dB mean | pinned between 3.0 and 7.0 dB; mid-column within 1.0 dB of Kraken | `test_benchmarks_published.py::test_bellhop_loses_the_near_bed_field_of_pe_workshop_3b` |

The wave engines' nulls all land within 30 m of the table's, 10 m short in
all three, so that shift is the reference's. Null control: a lossless bottom,
11 dB median from the reference
(`test_benchmarks_published.py::test_wave_engines_match_the_pe_workshop_3b_reference`).

### 3.3 Square-wave corrugated seafloor, 25 Hz

100 m of water over a 1704.5 m/s, density 2.5 bottom, with a 10 m × 100 m
square-wave corrugation from 5 to 10 km; source 18 m, receiver 50 m.
Reference: COUPLE07 Test Case 3, the fully two-way solution,
`benchmarks/corrugated_seafloor_25hz_couple07.txt` (Evans & Gilbert, JASA 77,
983-988, 1985); a density-1.5 run of the same input is
`benchmarks/corrugated_seafloor_rho15_25hz_couple07.txt`
(`test_benchmarks_published.py::test_the_corrugated_seafloor_reference_table_is_the_published_run`,
`test_benchmarks_published.py::test_the_density_1p5_corrugation_table_is_the_published_run_with_one_change`).

The measure is the mean TL offset from the two-way reference over
10-18.09 km, pinned on both sides: a change either way is news
(`test_benchmarks_published.py::test_one_way_engines_lose_more_to_the_corrugation_than_the_two_way_reference`).

| Engine | Mean offset | Pinned within | Null control (flat guide) | Test |
|---|---|---|---|---|
| RAM mpirams, density 2.5 | +4.35 dB | (3.0, 5.7) | -1.95 dB | `test_benchmarks_published.py::test_one_way_engines_lose_more_to_the_corrugation_than_the_two_way_reference` |
| RAM ramgeo, density 2.5 | +5.47 dB | (4.0, 7.0) | outside | `test_benchmarks_published.py::test_one_way_engines_lose_more_to_the_corrugation_than_the_two_way_reference` |
| Kraken coupled, density 2.5 | +1.20 dB | (0.3, 2.5) | -1.97 dB | `test_benchmarks_published.py::test_one_way_engines_lose_more_to_the_corrugation_than_the_two_way_reference` |
| RAM mpirams, density 1.5 | +0.50 dB | (-0.3, 1.3) | -1.03 dB | `test_benchmarks_published.py::test_the_pe_offset_beyond_the_corrugation_grows_with_the_bottom_density` |

Why the one-way engines differ is in §5.

### 3.4 Bucker fluid waveguide, 100 Hz

Water 240 m deep with a weak sound-speed minimum (1500, 1498, 1500 m/s at 0,
120, 240 m) over a 1505 m/s, density 2.1 lossless half-space; source 30 m,
receiver 90 m (*Computational Ocean Acoustics*, Sect. 4.10.2,
Figs. 4.17-4.18). The weak speed contrast traps few modes while the density
contrast puts much of the field in the continuous spectrum. Reference:
Scooter with `c_high=1e9` through uacpy, `benchmarks/bucker_waveguide_100hz_scooter.txt`,
checked against OAST
(`test_benchmarks_published.py::test_the_bucker_reference_table_spans_the_published_ranges`).

| Engine | Median / p90 TL difference over 2-20 km | Bound | Test |
|---|---|---|---|
| OAST at its defaults | 0.006 / 0.03 dB | 0.05 / 0.25 dB | `test_benchmarks_published.py::test_two_wavenumber_integration_codes_agree_on_the_bucker_waveguide` |
| Kraken `leaky_modes=True` | 0.013 / 0.06 dB | 0.05 / 0.3 dB | `test_benchmarks_published.py::test_engines_reproduce_the_bucker_waveguide` |
| Kraken at its defaults | 0.128 / 1.71 dB | 0.3 / 2.3 dB | `test_benchmarks_published.py::test_engines_reproduce_the_bucker_waveguide` |
| Scooter at its defaults | 0.102 / 0.64 dB | 0.3 / 1.2 dB | `test_benchmarks_published.py::test_engines_reproduce_the_bucker_waveguide` |
| RAM `zmax=2000` | 0.085 / 0.62 dB | 0.3 / 1.2 dB | `test_benchmarks_published.py::test_engines_reproduce_the_bucker_waveguide` |
| RAM at its defaults (automatic domain, zmax 1659 m) | 0.121 dB median | < 0.3 dB | `test_benchmarks_published.py::test_ram_default_domain_reaches_the_bucker_continuous_spectrum` |
| Kraken trapped modes only (`c_high` at 1505 m/s) | 1.542 / 8.43 dB | pinned between 0.8 and 2.5 dB median, p90 > 4.0 | `test_benchmarks_published.py::test_a_trapped_mode_sum_misses_the_bucker_continuous_spectrum` |

Null control: a density-1.0 half-space, 2.4-2.7 dB median from the reference
(`test_benchmarks_published.py::test_engines_reproduce_the_bucker_waveguide`).

### 3.5 NORDA PE Workshop I, Test Case 4c, 25 Hz

3410 m of deep water to 150 km, an upslope to 200 m by 200 km and a 200 m
shelf to 250 km, over a 454 m sediment that follows the seafloor; source
600 m, receivers 150 and 700 m. Reference: COUPLE07 Test Case 4,
`benchmarks/pe_workshop_4c_25hz_couple07.txt`
(`test_benchmarks_published.py::test_the_pe_workshop_4c_reference_table_spans_the_published_ranges`).
COUPLE makes `1/c²` linear between the listed points, and Kraken fed the same
points `c`-linear sat 3.7 dB median off where `1/c²`-linear gives 0.35 dB
(`benchmarks/pe_workshop_4c_25hz_couple07.txt`). The engine tests here are marked `slow`.

| Engine | Flat 5-150 km, median / p90 at 150 m and 700 m | Slope / shelf RMS at 150 m | Bounds (flat median, flat p90, slope, shelf) | Test |
|---|---|---|---|---|
| RAM (automatic grid) | 0.64 / 2.41 and 0.83 / 3.03 dB | 0.44 / 0.74 dB | 1.0, 3.6, 1.0, 1.3 | `test_benchmarks_published.py::test_engines_follow_pe_workshop_4c_up_the_slope_and_onto_the_shelf` |
| Kraken adiabatic | 0.35 / 1.44 and 0.36 / 1.27 dB | 1.27 / 1.44 dB | 0.6, 2.0, 2.0, 2.2 | `test_benchmarks_published.py::test_engines_follow_pe_workshop_4c_up_the_slope_and_onto_the_shelf` |
| Kraken coupled (automatic decomposition) | — | shelf 21.98 dB RMS, mean -21.67 dB | expected failure (strict) | `test_benchmarks_published.py::test_kraken_coupled_modes_keep_the_pe_workshop_4c_shelf_level` |

Null control: the slope removed, 4.5 dB RMS over the slope and 3.2 dB over
the shelf
(`test_benchmarks_published.py::test_engines_follow_pe_workshop_4c_up_the_slope_and_onto_the_shelf`).

---

## 4. What is not benchmarked

No reference was available to ship for the elastic wedges (Jensen, Nielsen,
Zampolli, Collins & Siegmann, 2007; the finite-element reference exists only
as figures), for the reverberation workshops, SWAM'99 and the 2024 soundscape
benchmark (inter-model comparisons with no reference solution, or a capability
uacpy lacks), or for any 3-D case (uacpy has no 3-D propagation route). An
elastic range-dependent seabed is therefore not covered by this page.

---

## 5. Known limits

Each limit below is measured by a test that pins it on both sides, so a fix
shows up as a failing test rather than a silently better number.

- **Bellhop at low `D/λ`.** A ray model needs many wavelengths of water
  ([Bellhop §2](bellhop.md#2-when-to-use-it--and-when-not-to)). On the ASA
  wedge at 25 Hz the water is 3.3 wavelengths deep and Bellhop sits at
  2.023 dB RMS, with a null the reference does not have, whatever the beam
  count
  (`test_benchmarks_published.py::test_bellhop_follows_the_asa_wedge_within_a_ray_models_error_at_25_hz`).
  Half a metre above the seafloor in Test Case 3b it is 5.37 dB median off,
  and against Kraken the gap closes to 0.44 dB as source and receiver rise to
  50 m (`test_benchmarks_published.py::test_bellhop_loses_the_near_bed_field_of_pe_workshop_3b`).
  On the stepped waveguide at 25 Hz its 1.233 dB median is the method's error
  (`test_benchmarks_coupled_mode.py::test_bellhop_stepped_waveguide_matches_coupled_mode_reference`).
- **The PE codes on a dense stair-step.** Over the corrugation of §3.3 RAM
  keeps the field across the vertical step faces, an amplitude error that
  grows with the density contrast (Collins & Siegmann, *Parabolic Wave
  Equations with Applications*, Sect. 2.6): mpirams is +4.35 dB at density
  2.5, +0.50 at 1.5 and -0.11 at 1.0; ramgeo is +5.47 at 2.5, and rams, the
  elastic PE with its own energy-flux correction, +5.03 to +5.41 over a 50 m/s
  shear bottom; the mpirams figure does not move with the grid (+4.26 to +4.48 dB over dr, dz and
  Padé order)
  (`test_benchmarks_published.py::test_one_way_engines_lose_more_to_the_corrugation_than_the_two_way_reference`,
  `test_benchmarks_published.py::test_the_pe_offset_beyond_the_corrugation_grows_with_the_bottom_density`;
  [RAM §9](ram.md#9-gotchas)).
- **Kraken coupled modes on Test Case 4c.** At its automatic decomposition
  `Kraken(mode_coupling='coupled')` puts the shelf 21.98 dB RMS from the
  reference, the field gaining energy; the test is a strict expected failure
  (`test_benchmarks_published.py::test_kraken_coupled_modes_keep_the_pe_workshop_4c_shelf_level`).
  `field.exe`'s projection amplifies modes at and above the half-space speed,
  and every coupled run also sums its own modes adiabatically and warns when
  the two far fields differ by more than 10 dB; this run warns at 27.5 dB
  ([Kraken §6.6](kraken.md#66-range-dependence-adiabatic-versus-coupled)).
- **The coupled method's own error.** Where the projection behaves, what is
  left is pressure matching itself: COUPLE07 run with the same one-way
  pressure matching sits 4.1 dB below its single-scatter reference on the 4c
  shelf at 150 m, and on the corrugation it is 1.13 dB above its two-way
  solution where Kraken's coupled run is at 1.20 dB
  ([Kraken §6.6](kraken.md#66-range-dependence-adiabatic-versus-coupled)).
  On the ASA wedge the coupled answer moves with the segment count rather
  than converging: 2.747, 2.908 and 3.496 dB at 30 m for 15, 51 and 200
  profiles
  (`test_benchmarks_published.py::test_kraken_mode_sums_follow_the_asa_wedge_within_their_method_error`).
- **Kraken's trapped modes on Bucker's waveguide.** A sum over the trapped
  modes misses the continuous spectrum a dense bottom feeds: 1.542 dB median,
  8.43 dB p90, against 0.013 / 0.06 dB with `leaky_modes=True`; the defaults
  sit between, at 0.128 / 1.71 dB
  (`test_benchmarks_published.py::test_a_trapped_mode_sum_misses_the_bucker_continuous_spectrum`,
  `test_benchmarks_published.py::test_engines_reproduce_the_bucker_waveguide`;
  [Kraken §6.5](kraken.md#65-leaky-modes)).

---

## 6. Running them

Every benchmark carries the `benchmark` marker; those that run an engine also
carry `requires_binary` (the OASES ones `requires_oases` too), and the
Test Case 4c group and the density sweep are `slow`.

```bash
# every benchmark
uacpy_venv/bin/python -m pytest uacpy/tests -m benchmark

# without the slow ones (Test Case 4c, the density sweep)
uacpy_venv/bin/python -m pytest uacpy/tests -m "benchmark and not slow"

# the ones that need no native binary: formulas, reference self-checks, tables
uacpy_venv/bin/python -m pytest uacpy/tests -m "benchmark and not requires_binary"

# this page's gate
uacpy_venv/bin/python -m pytest uacpy/tests/test_validation_page.py
```

A binary that is not built turns its tests into skips; pass `-rs` to see
them. The benchmark modules are `uacpy/tests/test_benchmarks_*.py`, and the
reference tables `uacpy/tests/data/benchmarks/*.txt`.
