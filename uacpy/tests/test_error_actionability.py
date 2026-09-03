"""What a failed call tells the user: the offending value, and the next action.

Two separate properties, measured and fixed as two axes:

1. **The message names the offending runtime value.** "pf must be in (0, 1)"
   states the contract but leaves the caller diffing their own inputs to find
   which one broke it. The value belongs in the message.
2. **Something names a concrete next action** — a parameter to change, a range,
   an option letter to add to a model deck. It may sit in the message body or
   in the typed ``remediation=`` field; both render, and which one carries it
   is a formatting choice, not an information one.

The sites pinned here were the ones that had *neither*, plus the readers whose
"file not found" said nothing about what writes the file. Each test names one
value or one action that the message has to carry, so rewording is free and
dropping the information is not.
"""

import io
import re

import numpy as np
import pytest

from uacpy.core.exceptions import ConfigurationError, FileFormatError


class TestReadersNameWhatWritesTheFile:
    """A missing model output is not the user's file: they cannot create it by
    hand, so "not found" alone leaves them nowhere to go. Each reader names the
    option letter whose absence is the usual reason the file was never
    written."""

    def test_oasn_covariance_not_found_names_the_option_letter(self, tmp_path):
        from uacpy.io.oases_reader import read_oasn_covariance
        with pytest.raises(FileFormatError) as exc:
            read_oasn_covariance(tmp_path / 'absent.xsm')
        assert "'N'" in exc.value.remediation
        assert '.xsm' in exc.value.remediation

    def test_oasn_replicas_not_found_names_the_option_letter(self, tmp_path):
        from uacpy.io.oases_reader import read_oasn_replicas
        with pytest.raises(FileFormatError) as exc:
            read_oasn_replicas(tmp_path / 'absent.rpo')
        assert "'R'" in exc.value.remediation
        assert '.rpo' in exc.value.remediation

    def test_oasr_reflection_not_found_names_the_option_letter(self, tmp_path):
        from uacpy.io.oases_reader import read_oasr_reflection_coefficients
        with pytest.raises(FileFormatError) as exc:
            read_oasr_reflection_coefficients(tmp_path / 'absent.rco')
        assert "'T'" in exc.value.remediation

    def test_oasp_trf_not_found_sends_the_user_to_the_run_log(self, tmp_path):
        from uacpy.io.oases_reader import read_oasp_trf
        with pytest.raises(FileFormatError) as exc:
            read_oasp_trf(tmp_path / 'absent.trf', receiver_depths=[10.0])
        assert 'stdout' in exc.value.remediation

    def test_rhs_not_found_names_both_reasons_a_run_writes_none(self, tmp_path):
        """The option letter is only half of it: SCTRHS also skips a smooth
        interface, so a deck with 's' set and no roughness still writes
        nothing. A remediation naming only the letter sends that user in a
        circle."""
        from uacpy.io.oases_reader import read_oases_rhs_header
        with pytest.raises(FileFormatError) as exc:
            read_oases_rhs_header(tmp_path / 'absent.rhs')
        assert "'s'" in exc.value.remediation
        assert 'rough' in exc.value.remediation.lower()


class TestTruncatedFortranRecordsNameTheFile:
    """``read_fortran_record`` is shared by every binary reader, so its EOF
    message was the one diagnostic a user saw with no idea which file it came
    from."""

    def test_eof_on_the_record_head_names_the_file_and_offset(self, tmp_path):
        from uacpy.io._fortran_helpers import read_fortran_record
        p = tmp_path / 'truncated.trf'
        p.write_bytes(b'\x01\x02')
        with open(p, 'rb') as f:
            with pytest.raises(FileFormatError) as exc:
                read_fortran_record(f, 'i')
        assert 'truncated.trf' in str(exc.value)
        assert 'byte 2' in str(exc.value)

    def test_eof_on_the_record_tail_names_the_payload_it_had_read(self, tmp_path):
        p = tmp_path / 'shorttail.trf'
        # A complete 4-byte head and payload, then a truncated tail marker.
        p.write_bytes((4).to_bytes(4, 'little') + b'abcd' + b'\x00')
        from uacpy.io._fortran_helpers import read_fortran_record
        with open(p, 'rb') as f:
            with pytest.raises(FileFormatError) as exc:
                read_fortran_record(f, 'i')
        assert 'shorttail.trf' in str(exc.value)
        assert '4-byte payload' in str(exc.value)

    def test_a_complete_record_reads_back_its_payload(self, tmp_path):
        """The other side of the guard: the EOF checks must not fire on a
        well-formed record, or every reader in the package would raise."""
        p = tmp_path / 'whole.trf'
        p.write_bytes((4).to_bytes(4, 'little') + (7).to_bytes(4, 'little')
                      + (4).to_bytes(4, 'little'))
        from uacpy.io._fortran_helpers import read_fortran_record
        with open(p, 'rb') as f:
            assert read_fortran_record(f, 'i') == (7,)


class TestEmptyInputNamesBothLengths:
    """"empty input" does not say *which* argument was empty. These carry both
    sizes, so the caller reads the answer instead of bisecting."""

    def test_bit_error_rate_names_both_stream_lengths(self):
        from uacpy.comms.metrics import bit_error_rate
        with pytest.raises(ConfigurationError) as exc:
            bit_error_rate([], [1, 0, 1])
        assert '0 bits' in str(exc.value)
        assert '3' in str(exc.value)

    def test_symbol_error_rate_names_both_stream_lengths(self):
        from uacpy.comms.metrics import symbol_error_rate
        with pytest.raises(ConfigurationError) as exc:
            symbol_error_rate([1, 2], [])
        assert '2 symbols' in str(exc.value)

    def test_evm_names_both_stream_lengths(self):
        from uacpy.comms.metrics import evm
        with pytest.raises(ConfigurationError) as exc:
            evm([], [1 + 0j])
        assert '0 symbols' in str(exc.value)

    def test_ofdm_demodulate_names_the_block_length_it_needed(self):
        from uacpy.comms.ofdm import ofdm_demodulate
        with pytest.raises(ConfigurationError) as exc:
            ofdm_demodulate(np.zeros(3, dtype=complex), n_subcarriers=8,
                            cp_len=2)
        msg = str(exc.value)
        assert '3 samples' in msg
        assert '8 + 2 = 10' in msg

    def test_matched_filter_metric_names_both_lengths(self):
        from uacpy.comms.sync import matched_filter_metric
        with pytest.raises(ConfigurationError) as exc:
            matched_filter_metric(np.zeros(4), np.zeros(9))
        msg = str(exc.value)
        assert '9 samples' in msg
        assert '4' in msg

    def test_spread_on_an_empty_code_names_a_generator_to_call(self):
        from uacpy.comms.dsss import spread
        with pytest.raises(ConfigurationError) as exc:
            spread([1, -1], [])
        assert 'm_sequence' in str(exc.value)

    def test_apply_channel_on_an_empty_h_names_the_identity_channel(self):
        from uacpy.comms.channel_models import apply_channel
        with pytest.raises(ConfigurationError) as exc:
            apply_channel(np.zeros(4), [])
        assert 'h=[1.0]' in str(exc.value)

    def test_run_parallel_on_no_jobs_names_the_job_constructor(self):
        from uacpy.parallel import run_parallel
        with pytest.raises(ConfigurationError) as exc:
            run_parallel([])
        assert 'Job(' in str(exc.value)


class TestConstraintMessagesNameTheOffendingValue:
    """Each of these stated its contract and not the input that broke it."""

    def test_probability_of_detection_names_the_out_of_range_pf(self):
        from uacpy.sonar.detection import probability_of_detection
        with pytest.raises(ConfigurationError) as exc:
            probability_of_detection(3.0, pf=1.5)
        assert '1.5' in str(exc.value)

    def test_probability_of_detection_accepts_pf_inside_the_interval(self):
        """The open interval is the boundary this guard defends; a pf just
        inside it must pass, or the message above would be unreachable."""
        from uacpy.sonar.detection import probability_of_detection
        assert np.isfinite(probability_of_detection(3.0, pf=1e-6))

    def test_albersheim_snr_names_the_offending_pulse_count(self):
        from uacpy.sonar.detection import albersheim_snr
        with pytest.raises(ConfigurationError) as exc:
            albersheim_snr(0.5, 1e-4, n_pulses=0)
        assert 'got 0' in str(exc.value)

    def test_column_scattering_strength_names_the_offending_thickness(self):
        from uacpy.sonar.scattering import column_scattering_strength
        with pytest.raises(ConfigurationError) as exc:
            column_scattering_strength(-30.0, thickness_m=0.0)
        assert 'got 0.0' in str(exc.value)

    def test_detection_threshold_energy_names_both_offending_inputs(self):
        from uacpy.sonar.detection import detection_threshold_energy
        with pytest.raises(ConfigurationError) as exc:
            detection_threshold_energy(0.5, 1e-4, bandwidth_hz=-1.0,
                                       integration_time_s=2.0)
        msg = str(exc.value)
        assert 'bandwidth_hz=-1.0' in msg
        assert 'integration_time_s=2.0' in msg

    def test_rrc_filter_names_the_offending_rolloff(self):
        from uacpy.comms.phy import rrc_filter
        with pytest.raises(ConfigurationError) as exc:
            rrc_filter(sps=4, rolloff=1.5, span=8)
        assert '1.5' in str(exc.value)

    def test_janus_app_type_names_the_offending_value(self):
        from uacpy.comms.janus import JanusPacket
        pkt = JanusPacket(class_id=16, app_type=99,
                          app_data=np.zeros(34, dtype=int))
        with pytest.raises(ConfigurationError) as exc:
            pkt.to_bits()
        assert '99' in str(exc.value)

    def test_janus_app_data_names_the_length_it_got(self):
        from uacpy.comms.janus import JanusPacket
        pkt = JanusPacket(class_id=16, app_type=1,
                          app_data=np.zeros(10, dtype=int))
        with pytest.raises(ConfigurationError) as exc:
            pkt.to_bits()
        assert 'got 10' in str(exc.value)

    def test_omp_estimate_names_both_ends_of_the_interval(self):
        from uacpy.comms.channel_est import omp_estimate
        with pytest.raises(ConfigurationError) as exc:
            omp_estimate(np.zeros(8, dtype=complex), np.ones(8), n_taps=4,
                         sparsity=9)
        msg = str(exc.value)
        assert 'sparsity=9' in msg
        assert 'n_taps=4' in msg

    def test_impulse_response_names_both_shapes(self):
        from uacpy.acoustic_signal.channel import (
            impulse_response_from_transfer_function)
        with pytest.raises(ConfigurationError) as exc:
            impulse_response_from_transfer_function(
                np.ones(3, dtype=complex), np.array([1.0, 2.0]),
                sample_rate=1000.0)
        msg = str(exc.value)
        assert '(2,)' in msg
        assert '(3,)' in msg


class TestJanusSeparatesTheDopplerStageFromDetection:
    """Both raises said "preamble not found", so a user could not tell a
    waveform carrying no packet from one whose preamble the Doppler
    correction lost. The second names the stage and the scale it applied."""

    def test_no_preamble_at_all_names_the_band_and_the_duration(self):
        from uacpy.comms.janus import janus_demodulate
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(4096)
        with pytest.raises(ConfigurationError) as exc:
            janus_demodulate(noise, 44100.0, doppler_max_speed=0.0)
        msg = str(exc.value)
        assert 'start=' in msg
        assert 'Hz' in msg
        assert 'after Doppler correction' not in msg

    def test_a_preamble_lost_only_after_resampling_names_that_stage(self,
                                                                    monkeypatch):
        """Reaching the second raise needs a detector that succeeds and then
        fails, which no fixed waveform reliably produces — so the two
        ``_detect`` outcomes are supplied directly. What is under test is
        which of the two messages the caller gets, not the detector."""
        from uacpy.comms import janus

        calls = []

        def fake_detect(*args, **kwargs):
            calls.append(1)
            return (0, None) if len(calls) == 1 else (None, None)

        monkeypatch.setattr(janus, '_detect', fake_detect)
        monkeypatch.setattr(janus, '_estimate_doppler',
                            lambda *a, **k: 1.002)
        rng = np.random.default_rng(0)
        with pytest.raises(ConfigurationError) as exc:
            janus.janus_demodulate(rng.standard_normal(8192), 44100.0,
                                   doppler_max_speed=5.0)
        msg = str(exc.value)
        assert 'after Doppler correction' in msg
        assert '1.002' in msg
        assert 'doppler_max_speed=5.0' in msg
        assert len(calls) == 2


class TestVisualizationNamesWhatToSupply:

    def test_compare_on_an_empty_list_names_the_cut_it_needs(self):
        from uacpy.visualization.plots.fields import compare
        with pytest.raises(ConfigurationError) as exc:
            compare([])
        assert 'field.at' in str(exc.value)

    def test_compare_models_names_both_accepted_shapes(self):
        from uacpy.visualization.plots.fields import compare_models
        with pytest.raises(ConfigurationError) as exc:
            compare_models([])
        assert 'dict' in str(exc.value)


def _inline_remediations():
    """``([(site, text)], indirect count)`` for every ``remediation=`` written
    inline at a ``raise`` in shipped code.

    A site passing a variable (``remediation=remedy``) builds its text
    elsewhere, so the literal is not visible here; those are counted rather
    than read, and the count is pinned by the caller so the blind spot cannot
    grow unremarked."""
    import ast
    from pathlib import Path

    pkg = Path(__file__).resolve().parent.parent
    inline = (ast.Constant, ast.JoinedStr, ast.BinOp)
    found, indirect = [], 0
    for path in sorted(pkg.rglob('*.py')):
        if 'tests' in path.parts or 'third_party' in path.parts:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or not isinstance(
                    node.exc, ast.Call):
                continue
            for kw in node.exc.keywords:
                if kw.arg != 'remediation':
                    continue
                if not isinstance(kw.value, inline):
                    indirect += 1
                    continue
                text = ' '.join(
                    n.value for n in ast.walk(kw.value)
                    if isinstance(n, ast.Constant)
                    and isinstance(n.value, str))
                found.append(
                    (f'{path.relative_to(pkg)}:{node.lineno}', text))
    return found, indirect


def test_no_inline_remediation_says_only_to_check_your_input():
    """A remediation that does not name an action is worse than none: it
    renders a "How to fix:" heading over advice the user cannot act on.

    Scope: remediations written *inline* at the raise site — see
    ``_inline_remediations``. Nine sites build their text elsewhere and are out
    of that sweep's reach rather than clean; it reports what it can see, and
    the tables it cannot are read directly below.
    """
    remediations, indirect = _inline_remediations()
    # Copernicus keeps three per-fetcher wordings in a module-level table
    # rather than at three copies of one raise, so read them from the table.
    from uacpy.data.copernicus import _DATE_GAP_MESSAGES
    remediations = list(remediations) + [
        (f'data/copernicus.py:_DATE_GAP_MESSAGES[{kind!r}]', text)
        for kind, (_message, text) in _DATE_GAP_MESSAGES.items()]
    empty = [f'{site}: {text!r}' for site, text in remediations
             if text.strip().lower().rstrip('.') in (
                 'check your input', 'check your inputs', 'check the input',
                 'see the documentation', 'try again', '')]
    assert not empty, 'contentless remediation(s):\n' + '\n'.join(empty)
    # Pins the blind spot itself: if indirection spreads, this count moves and
    # the gate's coverage claim has to be restated rather than silently
    # shrinking. It moved 8 -> 9 when the three Copernicus date-gap guards
    # collapsed into one helper reading _DATE_GAP_MESSAGES; that table is read
    # above, so the gate sees those three remediations, not none.
    assert indirect <= 9, f'{indirect} remediations now built indirectly'


def test_no_remediation_offers_only_the_editable_install_command():
    """``pip install -e ".[copernicus]"`` needs a checkout to point ``.`` at.

    The lens is a user who installed the distribution: for them the editable
    form has no target, so a remediation that names only it is an instruction
    they cannot run. Naming the non-editable form as well costs one clause,
    and the four sibling optional-dependency remediations in ``uacpy/data``
    already carry it (``pip install pyproj``, ``pip install netCDF4``, …).
    """
    remediations, _ = _inline_remediations()
    editable_only = [
        f'{site}: {text!r}' for site, text in remediations
        if 'pip install -e' in text
        and not re.search(r'pip install (?!-e)[\'"`\w]', text)]
    assert not editable_only, (
        'remediation(s) offering only the editable-from-a-checkout install '
        'command, which a wheel/sdist user cannot run:\n'
        + '\n'.join(editable_only))


def _time_domain_field():
    """A Field on a time axis — not transmission loss."""
    from uacpy.core.results.field import Field
    return Field(data=np.zeros((3, 4)),
                 coords={'time': np.arange(3.0), 'range': np.arange(4.0)})


def _complex_se_field():
    """A complex Field where signal excess must be real dB."""
    from uacpy.core.results.field import Field
    return Field(data=np.zeros((3, 4), dtype=complex),
                 coords={'depth': np.arange(3.0), 'range': np.arange(4.0)})


def _pressure_slab(depth, range_):
    from uacpy.core.results.field import Field
    return Field(data=np.ones((depth.size, range_.size), dtype=complex),
                 coords={'depth': depth, 'range': range_})


def _stack_of_slabs(depths):
    """A source-depth ResultStack whose slabs sit on the given depth axes."""
    from uacpy.core.results.field import ResultStack
    r = np.linspace(500, 4000, 6)
    slabs = [_pressure_slab(z, r) for z in depths]
    return ResultStack(slabs, np.arange(float(len(depths))),
                       coordinate_name='source_depth')


class _FlatModes:
    """Minimal KRAKEN-shaped modes: two modes, flat shapes, real wavenumbers."""
    k = np.array([0.1 + 0j, 0.2 + 0j])
    z = np.linspace(0.0, 100.0, 11)
    phi = np.ones((11, 2))


def _guard_cases():
    """``(id, call, fragments)`` for every guard rewritten to name its input.

    Built lazily inside the function so the module-level import list stays
    short; each entry triggers exactly one raise.
    """
    from uacpy.acoustic_signal.active import (
        ambiguity_function, matched_filter, processing_gain)
    from uacpy.acoustic_signal.analysis import sel
    from uacpy.acoustic_signal.arrays import sample_covariance
    from uacpy.acoustic_signal.bands import (
        decidecade_band_levels, decidecade_bands)
    from uacpy.acoustic_signal.channel import impulse_response
    from uacpy.acoustic_signal.constant_q import (
        constant_q_spectrogram, constant_q_transform)
    from uacpy.acoustic_signal.modal import modal_group_velocity
    from uacpy.acoustic_signal.noise_synthesis import (
        synthesize_noise_from_psd)
    from uacpy.acoustic_signal.sequences import bpsk_modulate, mseq
    from uacpy.acoustic_signal.timefreq import (
        ComplexCepstrum, analytic_signal, cepstrum, complex_cepstrum, cwt,
        inverse_complex_cepstrum, inverse_cwt, wigner_ville)
    from uacpy.acoustic_signal.transforms import (
        fk_transform, inverse_fk, inverse_radon, inverse_taup,
        radon_transform, taup_transform)
    from uacpy.comms.janus import janus_decode
    from uacpy.comms.transceiver import OFDMReceiver, OFDMTransmitter
    from uacpy.sonar.matched_field import (
        replica_bank_from_field, synthesize_replica)
    from uacpy.sonar.reverberation import total_reverberation
    from uacpy.sonar.sonar_equation import (
        detection_range, noise_background, passive_signal_excess_field,
        probability_of_detection_field)

    z4 = np.linspace(5, 95, 4)
    return [
        # ── acoustic_signal/transforms.py ────────────────────────────────
        ('fk_transform_window_list_length',
         lambda: fk_transform(np.zeros((8, 4)), 1000.0, 1.0, window=['hann']),
         ('got 1 entries',)),
        ('fk_transform_nfft_tuple_length',
         lambda: fk_transform(np.zeros((8, 4)), 1000.0, 1.0, nfft=(8, 4, 2)),
         ('got 3 entries',)),
        ('radon_transform_data_ndim',
         lambda: radon_transform(np.zeros(8), 1000.0, 1.0, [1.0]),
         ('got shape (8,)',)),
        ('inverse_radon_R_ndim',
         lambda: inverse_radon(np.zeros(8), 1000.0, 1.0, [1.0], 4),
         ('got shape (8,)',)),
        ('taup_transform_data_ndim',
         lambda: taup_transform(np.zeros(8), 1000.0, 1.0),
         ('got shape (8,)',)),
        ('inverse_taup_taup_ndim',
         lambda: inverse_taup(np.zeros(8), [1.0], 1000.0, 1.0, 4),
         ('got shape (8,)',)),
        ('inverse_fk_FK_ndim',
         lambda: inverse_fk(np.zeros(8)),
         ('got shape (8,)',)),
        ('fk_transform_data_ndim',
         lambda: fk_transform(np.zeros(8), 1000.0, 1.0),
         ('got shape (8,)',)),
        # ── acoustic_signal/timefreq.py ──────────────────────────────────
        ('analytic_signal_data_ndim',
         lambda: analytic_signal(np.zeros((2, 3))),
         ('got shape (2, 3)',)),
        ('wigner_ville_window_length',
         lambda: wigner_ville(np.zeros(32), 1000.0, time_window=0),
         ('got 0',)),
        ('wigner_ville_window_ndim',
         lambda: wigner_ville(np.zeros(32), 1000.0,
                              time_window=np.zeros((2, 3))),
         ('got shape (2, 3)',)),
        ('wigner_ville_data_ndim',
         lambda: wigner_ville(np.zeros((2, 3)), 1000.0),
         ('got shape (2, 3)',)),
        ('cwt_data_ndim',
         lambda: cwt(np.zeros((2, 3)), 1000.0),
         ('got shape (2, 3)',)),
        ('cwt_frequencies_positive',
         lambda: cwt(np.zeros(64), 1000.0, frequencies=[10.0, 0.0, -5.0]),
         ('got 2 value(s) <= 0', 'first at index 1')),
        ('inverse_cwt_W_shape',
         lambda: inverse_cwt(np.zeros((2, 4)), [1.0, 2.0, 3.0], 1000.0),
         ('got W shape (2, 4)', '3 frequencies')),
        ('cepstrum_data_ndim',
         lambda: cepstrum(np.zeros((2, 3))),
         ('got shape (2, 3)',)),
        ('complex_cepstrum_data_ndim',
         lambda: complex_cepstrum(np.zeros((2, 3))),
         ('got shape (2, 3)',)),
        ('inverse_complex_cepstrum_type',
         lambda: inverse_complex_cepstrum(np.zeros(8)),
         ('Got ndarray',)),
        ('inverse_complex_cepstrum_ndim',
         lambda: inverse_complex_cepstrum(
             ComplexCepstrum(cepstrum=np.zeros((2, 3)), delay=0)),
         ('got shape (2, 3)',)),
        # ── acoustic_signal/active.py ────────────────────────────────────
        ('matched_filter_input_ndim',
         lambda: matched_filter(np.zeros((2, 3)), np.zeros(4)),
         ('received shape (2, 3)', 'replica shape (4,)')),
        ('processing_gain_bt_product',
         lambda: processing_gain(0.0, 2.0),
         ('bandwidth_hz=0.0', 'duration_s=2.0')),
        ('ambiguity_function_waveform_ndim',
         lambda: ambiguity_function(np.zeros((2, 3)), 1000.0),
         ('got shape (2, 3)',)),
        # ── acoustic_signal/analysis.py ──────────────────────────────────
        ('sel_no_samples_to_integrate',
         lambda: sel(np.ones(100), 1000.0, integration_time=1e-6),
         ('Got 0 sample(s)', 'integration_time=1e-06')),
        ('sel_no_band_below_nyquist',
         lambda: sel(np.ones(4096), 100.0, fmin=1000.0, fmax=2000.0),
         ('got fmin=1000.0', 'fmax=2000.0')),
        # ── acoustic_signal/bands.py ─────────────────────────────────────
        ('decidecade_bands_edges',
         lambda: decidecade_bands(0.0, 100.0),
         ('got f_low=0.0', 'f_high=100.0')),
        ('decidecade_band_levels_negative_psd',
         lambda: decidecade_band_levels(np.array([1.0, -2.0, -3.0]),
                                        np.array([10.0, 20.0, 30.0])),
         ('Got 2 negative value(s)', 'minimum -3')),
        ('decidecade_band_levels_frequency_order',
         lambda: decidecade_band_levels(np.array([1.0, 2.0, 3.0]),
                                        np.array([10.0, 30.0, 20.0])),
         ('Got 1 non-increasing step(s)', 'first at index 1')),
        # ── acoustic_signal/channel.py ───────────────────────────────────
        ('impulse_response_shapes',
         lambda: impulse_response(np.ones(3), np.zeros(4), 1000.0),
         ('amplitudes shape (3,)', 'delays_s shape (4,)')),
        ('impulse_response_negative_delays',
         lambda: impulse_response(np.ones(3), np.array([0.0, -1.0, -2.0]),
                                  1000.0),
         ('got 2 negative value(s)', 'first at index 1')),
        # ── acoustic_signal/constant_q.py ────────────────────────────────
        ('constant_q_hop',
         lambda: constant_q_spectrogram(np.zeros(4096), 8000.0, hop=0),
         ('got 0',)),
        ('constant_q_data_ndim',
         lambda: constant_q_transform(np.zeros((2, 3)), 8000.0),
         ('got shape (2, 3)',)),
        # ── acoustic_signal/modal.py ─────────────────────────────────────
        ('modal_group_velocity_frequency_order',
         lambda: modal_group_velocity([100.0, 50.0], [1.0, 2.0]),
         ('got shape (2,)', '1 non-increasing step(s)')),
        ('modal_group_velocity_wavenumber_rows',
         lambda: modal_group_velocity([1.0, 2.0, 3.0], np.zeros(2)),
         ('k_horizontal shape (2,)', '3 frequencies')),
        # ── acoustic_signal/noise_synthesis.py ───────────────────────────
        ('synthesize_noise_frequency_order',
         lambda: synthesize_noise_from_psd(np.ones(3),
                                           np.array([1.0, 3.0, 2.0])),
         ('got 1 non-increasing step(s)', 'first at index 1')),
        ('synthesize_noise_log_interp_positivity',
         lambda: synthesize_noise_from_psd(
             np.array([1.0, 0.0, 2.0]), np.array([1.0, 2.0, 3.0]),
             duration=0.01, interp='log'),
         ('Fxx[0]=1', 'non-positive Pxx value(s)')),
        # ── acoustic_signal/sequences.py ─────────────────────────────────
        ('bpsk_samples_per_chip_integer',
         lambda: bpsk_modulate(np.array([1, -1, 1]), 100.0, 1000.0, 300.0),
         ('sample_rate/chips_per_sec = 1000/300',)),
        ('mseq_register_length',
         lambda: mseq(20),
         ('got 20',)),
        # ── acoustic_signal/arrays.py ────────────────────────────────────
        ('sample_covariance_snapshots_ndim',
         lambda: sample_covariance(np.ones(8)),
         ('got shape (8,)',)),
        # ── acoustic_signal/_signal_validate.py ──────────────────────────
        ('require_finite_signal_nonfinite_count',
         lambda: analytic_signal(np.array([1.0, np.nan, 2.0, np.inf])),
         ('Got 2 non-finite value(s) of 4', 'first at flat index 1')),
        # ── comms ────────────────────────────────────────────────────────
        ('janus_decode_symbol_count',
         lambda: janus_decode(np.zeros(10)),
         ('got 10',)),
        ('ofdm_to_passband_band_fits',
         lambda: OFDMTransmitter('qpsk', n_subcarriers=16,
                                 cp_len=4).to_passband(
             np.zeros(64, dtype=complex), 8000.0, 100.0, oversample=1),
         ('got fc=100 Hz', 'sample_rate=8000 Hz', 'oversample=1')),
        ('ofdm_receiver_frame_length',
         lambda: OFDMReceiver('qpsk', n_subcarriers=16, cp_len=4).receive(
             np.zeros(20, dtype=complex)),
         ('block(s)', 'need >= 3')),
        # ── sonar ────────────────────────────────────────────────────────
        ('synthesize_replica_ranges_positive',
         lambda: synthesize_replica(_FlatModes(), 50.0,
                                    [1000.0, 0.0, -5.0], [10.0]),
         ('got 2 value(s) <= 0', 'first at index 1')),
        ('replica_bank_slab_depth_axis',
         lambda: replica_bank_from_field(_stack_of_slabs([z4]),
                                         array_depths=np.linspace(5, 95, 7)),
         ("slab 'depth' axis of shape (4,)", 'array_depths of shape (7,)')),
        ('replica_bank_slabs_share_axes',
         lambda: replica_bank_from_field(_stack_of_slabs([z4, z4 + 1.0])),
         ('Got slab 1',)),
        ('replica_bank_field_depth_axis',
         lambda: replica_bank_from_field(
             _pressure_slab(z4, np.linspace(500, 4000, 6)),
             array_depths=np.linspace(5, 95, 7)),
         ("field 'depth' axis of shape (4,)", 'array_depths of shape (7,)')),
        ('total_reverberation_zero_length',
         lambda: total_reverberation(np.array([1.0, 2.0]), np.array([])),
         ('Got sizes [2, 0]',)),
        ('noise_background_di_and_ag',
         lambda: noise_background(60.0, 10.0, array_gain=12.0),
         ('directivity_index=10.0', 'array_gain=12.0')),
        ('signal_excess_field_time_domain',
         lambda: passive_signal_excess_field(
             _time_domain_field(), source_level=180.0, noise_level=60.0,
             detection_threshold=10.0),
         ('Got axes [',)),
        ('pd_field_needs_real_db',
         lambda: probability_of_detection_field(_complex_se_field(),
                                                sigma_db=8.0),
         ('Got dtype complex',)),
        ('detection_range_shape_mismatch',
         lambda: detection_range(np.zeros(3), np.zeros(4)),
         ('ranges_m shape (3,)', 'signal_excess_db shape (4,)')),
    ]


_GUARD_CASES = _guard_cases()


class TestShapeAndArrayGuardsNameWhatTheyGot:
    """Fifty-one guards in ``acoustic_signal`` / ``comms`` / ``sonar`` stated
    their contract ("data must be 1-D", "delays_s must be >= 0") and left the
    caller to find the offending input by inspection.

    Each case names one fragment the message has to carry: a shape, a dtype, a
    scalar, or — for an array-valued input — a count of offenders with the
    first index. An array is reported by that checkable summary and never
    dumped: a message printing ten thousand floats is worse than one printing
    none.
    """

    @pytest.mark.parametrize(
        'call, fragments',
        [(call, frags) for _, call, frags in _GUARD_CASES],
        ids=[cid for cid, _, _ in _GUARD_CASES])
    def test_guard_names_the_offending_value(self, call, fragments):
        with pytest.raises(ConfigurationError) as exc:
            call()
        message = str(exc.value)
        for fragment in fragments:
            assert fragment in message, (
                f'{fragment!r} missing from: {message!r}')

    def test_the_pinned_set_does_not_shrink(self):
        """A parametrized pin is deleted by deleting one table row, which no
        other assertion here would notice — the suite would simply run fewer
        cases and stay green.

        A *fraction* gate over these packages was measured and rejected
        instead: every lexical proxy for "names the offending value" that can
        be computed (a ``got`` cue, an interpolation count, an interpolation
        overlapping the guard) mis-scores files that are already complete —
        ``comms/metrics.py`` names both stream lengths in all five of its
        messages and scores 0/5 on the ``got`` cue. A floor built on any of
        them flagged 22 of 28 files and would need a whitelist longer than the
        set it measures. This counts the one thing that can be counted
        exactly.
        """
        assert len(_GUARD_CASES) >= 51
        ids = [cid for cid, _, _ in _GUARD_CASES]
        assert len(set(ids)) == len(ids), 'duplicate case id'


def test_read_fortran_record_reports_a_stream_without_a_name():
    """The file name comes from ``f.name``, which a BytesIO does not have.
    The message must still render rather than raising AttributeError from
    inside the error path."""
    from uacpy.io._fortran_helpers import read_fortran_record
    with pytest.raises(FileFormatError) as exc:
        read_fortran_record(io.BytesIO(b'\x01'), 'i')
    assert '<stream>' in str(exc.value)
