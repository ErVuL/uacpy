"""
OASES - Ocean Acoustics and Seismic Exploration Synthesis

OASES is a comprehensive acoustic/seismic propagation modeling suite developed
by Henrik Schmidt at MIT. It includes multiple executables for different scenarios:

- **OAST**: Transmission Loss via wavenumber integration
- **OASN**: Noise covariance matrices and signal replicas (matched-field processing)
- **OASR**: Reflection coefficients at stratified interfaces
- **OASP**: Broadband transfer-function / pulse synthesis (wideband wavenumber integration)

This module provides Python wrappers for all OASES executables following
the UACPY propagation model architecture.

Usage
-----
```python
from uacpy.models import OAST, OASN, OASR, OASP

# Transmission loss using OAST
oast = OAST()
result = oast.run(env, source, receiver)

# Noise covariance / replicas using OASN
oasn = OASN()
cov = oasn.run(env, source, receiver, run_mode=RunMode.COVARIANCE)

# Reflection coefficients using OASR
oasr = OASR()
refl = oasr.run(env, source, receiver, run_mode=RunMode.REFLECTION)

# Broadband pulse synthesis using OASP (wavenumber integration)
oasp = OASP()
result = oasp.run(env, source, receiver, run_mode=RunMode.BROADBAND)
```
"""

import re
import warnings

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union

from uacpy.models.base import (
    PropagationModel, RunMode, ModelSpec, USER_FRAME_SKIP,
)
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import (
    Result, Field,
    Covariance, Replicas, ReflectionCoefficient,
)
from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError, ModelExecutionError,
    UnsupportedFeatureError,
)
from uacpy.io.oases_writer import write_oast_input, write_oasn_input, write_oasp_input, write_oasr_input
from uacpy.io.units import m_to_km
from uacpy.io.oases_reader import (
    read_oast_tl,
    read_oasn_covariance,
    read_oasn_replicas,
    read_oasp_trf,
    read_oasr_reflection_coefficients,
)


def _oases_resample_frequencies(
    freqs: np.ndarray, model_name: str,
) -> tuple[float, float, int, bool]:
    """Convert an arbitrary user ``frequencies=`` vector to the
    ``(fmin, fmax, N)`` triple OASR/OASP write into the input file.

    OASR's ``.dat`` and OASP's broadband kernel both express their
    frequency axis as a min/max/count, which implies equispaced
    sampling. If the user passes a non-equispaced vector we warn and
    auto-resample onto ``np.linspace(fmin, fmax, N)`` so the result's
    ``Result.frequencies`` reflects what the model actually saw.

    Returns
    -------
    fmin, fmax : float
    n : int
        Number of equispaced bins.
    resampled : bool
        True iff the user vector was non-equispaced and got resampled.
    """
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    fmin = float(freqs.min())
    fmax = float(freqs.max())
    n = int(freqs.size)
    resampled = False
    if n > 1:
        diffs = np.diff(freqs)
        # Equispaced if all diffs match the mean diff within tolerance
        # (rtol scaled to the band; atol is a tiny absolute floor).
        target = (fmax - fmin) / (n - 1)
        if not np.allclose(diffs, target, rtol=1e-6, atol=1e-9):
            resampled = True
            warnings.warn(
                f"{model_name}: frequencies= vector is non-equispaced; "
                f"OASES expresses the frequency axis as (fmin, fmax, N) "
                f"so the vector has been resampled onto "
                f"np.linspace({fmin}, {fmax}, {n}). "
                f"Result.frequencies will be the resampled grid, not "
                f"your input. Pass an equispaced vector (or "
                f"freq_min/freq_max/n_frequencies) to suppress.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
    return fmin, fmax, n, resampled


def _stack_oasr_data(data: dict):
    """Convert per-frequency lists from :func:`read_oasr_reflection_coefficients`
    into typed-Result arrays.

    Returns
    -------
    theta : ndarray, shape ``(n_angles,)``
    R : ndarray, shape ``(n_angles,)`` (single freq) or ``(n_angles, n_frequencies)``
    phi : ndarray, same shape as ``R``  — phase in radians (reader stores degrees)
    freqs : ndarray, shape ``(n_frequencies,)``
    """
    freqs = np.asarray(data.get('frequencies', []), dtype=float)
    angle_lists = data.get('angles_or_slowness', [])
    R_lists = data.get('magnitude', [])
    phi_lists = data.get('phase', [])
    if not angle_lists:
        raise ConfigurationError("OASR reader returned no frequency samples")
    theta = np.asarray(angle_lists[0], dtype=float)
    n_angles = len(theta)
    n_frequencies = len(freqs)
    for i, (a, m, p) in enumerate(zip(angle_lists, R_lists, phi_lists)):
        if len(a) != n_angles:
            raise ConfigurationError(
                f"OASR: angle grid mismatch between frequencies "
                f"(freq[{i}] has {len(a)} angles, freq[0] has {n_angles}). "
                f"Multi-frequency stacking requires a shared angle grid."
            )
        if len(m) != n_angles or len(p) != n_angles:
            raise ConfigurationError(
                f"OASR: payload length mismatch at frequency index {i}: "
                f"angles={len(a)}, magnitude={len(m)}, phase={len(p)}"
            )
    if n_frequencies == 1:
        R = np.asarray(R_lists[0], dtype=float)
        phi_deg = np.asarray(phi_lists[0], dtype=float)
    else:
        R = np.column_stack([np.asarray(m, dtype=float) for m in R_lists])
        phi_deg = np.column_stack([np.asarray(p, dtype=float) for p in phi_lists])
    return theta, R, np.deg2rad(phi_deg), freqs


def _oasn_freq_axis(data: dict) -> np.ndarray:
    """Reconstruct the frequency vector from an OASN reader payload.

    The OASN ``.xsm`` / ``.rpo`` headers carry ``freq_min``, ``freq_max``,
    ``n_frequencies`` (and a ``freq_delta`` cross-check). We rebuild the
    axis with ``np.linspace`` since the spacing OASN advertises is
    equispaced between FREQ1 and FREQ2.
    """
    n = int(data.get('n_frequencies', 1))
    f1 = float(data.get('freq_min', 0.0))
    f2 = float(data.get('freq_max', f1))
    if n <= 1:
        return np.array([f1], dtype=float)
    return np.linspace(f1, f2, n, dtype=float)


def _oases_find_executable(model: PropagationModel, name: str) -> Path:
    """Locate an OASES binary, preferring the ``<name>_bash`` wrapper.

    Searches ``uacpy/bin/oases``, ``uacpy/bin/oalib``, and
    ``uacpy/third_party/oases/bin``.
    """
    return model._find_executable_in_paths(
        [f'{name}_bash', name],
        bin_subdirs=['oases', 'oalib'],
        dev_subdir='oases',
    )


#: How much of a child stream to quote back in an error. OASES echoes the
#: whole environment and every wavenumber block to stdout, so only the tail
#: is useful; the abort reason is always last.
_OASES_STREAM_TAIL = 2000


def _tail(text: Optional[str], n_chars: int = _OASES_STREAM_TAIL) -> Optional[str]:
    """Last ``n_chars`` of a captured child stream, or ``None`` if empty."""
    if not text:
        return None
    return text if len(text) <= n_chars else '…' + text[-n_chars:]


def _stream_block(label: str, text: Optional[str]) -> str:
    """Render a captured child stream as a labelled block, or nothing."""
    trimmed = _tail(text)
    return f"\n\n{label}:\n{trimmed}" if trimmed else ''


#: OASES' wavenumber-array bound, ``NPEXP = 16, NP = 2**NPEXP``
#: (third_party/oases/src/compar.f:37-38). OAST stops above it
#: (unoast31.f:459 ``IF (NWVNO.GT.NP) STOP '>>> TOO MANY WAVENUMBERS <<<'``),
#: which is what catches its automatic branch — a pinned NW never reaches
#: that test, being clamped by ``NWVNO=MIN0(NWVNO,NP)`` at :435 first.
#: OASP has no such test at all — AUTSAM (unoasp22.f:1130
#: ``NW=(WNMAX-WNMIN)/DK+1``) applies no clamp and :174's MIN0 runs before
#: it — so it runs past the arrays and writes a transfer function whose
#: values are meaningless.
_OASES_NP = 65536

#: OASP echoes the count it settled on before the frequency loop
#: (unoasp22.f:340, :349).
_OASES_NWVNO_ECHO = re.compile(r'NO\. OF WAVENUMBERS:\s*(\d+)')


def _oases_subprocess_env(base_name: str, **extras: str) -> dict:
    """Build the FORnnn env-var dict the OASES csh wrappers set.

    OPFILW/OPFILR/OPFILB resolve every Fortran unit through
    ``GETENV('FORnnn')`` (oases/src/oashun21.f:555-556, :590, :629), falling
    back to ``<deck stem>.<nnn>`` when the variable is unset (:560, :633) —
    which is why the readers also look for a bare ``.023``. The binary is
    therefore run directly and this dict stands in for
    ``third_party/oases/bin/{oast,oasn,oasp,oasr}``. The keys every
    sub-model needs are FOR001 (the ``.dat`` deck), FOR019/FOR020 (the
    ``.plp`` header and ``.plt`` data OASES' plotters write), FOR028/FOR029
    (contour output, ``.cdr``/``.bdr`` in the wrappers) and FOR045
    (scattering right-hand sides, ``.rhs``); the last three are given inert
    numeric suffixes here because uacpy reads none of them. ``extras``
    supplies the per-binary units the wrappers add (e.g. FOR002='src' for
    OAST, FOR016='xsm' for OASN) — pass the suffix without the
    ``base_name + '.'`` prefix. ``base_name`` is the stem of the input file
    (no extension).
    """
    import os
    # ``base_name`` is interpolated straight into the FORnnn filenames the
    # OASES csh wrappers open; a value with a path separator or ``..`` would
    # become a traversal. It is a hard-coded literal in every caller today —
    # keep it that way by rejecting anything that isn't a plain stem.
    if not re.fullmatch(r'[A-Za-z0-9_]+', base_name):
        raise ConfigurationError(
            f"OASES base_name must match [A-Za-z0-9_]+; got {base_name!r}"
        )
    env = os.environ.copy()
    env['FOR001'] = f'{base_name}.dat'
    env['FOR019'] = f'{base_name}.plp'
    env['FOR020'] = f'{base_name}.plt'
    env['FOR028'] = f'{base_name}.028'
    env['FOR029'] = f'{base_name}.029'
    env['FOR045'] = f'{base_name}.045'
    for key, suffix in extras.items():
        env[key] = f'{base_name}.{suffix}'
    return env


class OASES(PropagationModel):
    """Public base + factory for the OASES sub-models (OAST/OASN/OASR/OASP).

    Each sub-model wraps one OASES binary and differs only in its per-binary
    ``FORnnn`` unit-number map (``_FOR_FILES``) and its run modes; all share the
    one ``_execute`` below. They subclass ``OASES``, so ``isinstance(model,
    OASES)`` is True for any of them. ``OASES`` itself is abstract — instantiate
    a sub-model directly, or use :meth:`OASES.for_mode` to pick one by
    ``RunMode``."""

    _FOR_FILES: dict = {}

    # Files this sub-model's binary writes: suffixes appended to
    # ``base_name``, plus the bare ``fort.NN`` names some OASES builds emit
    # instead. Both are cleared before launch so a pinned ``work_dir``
    # cannot return an earlier run's output as this run's answer.
    _OUTPUT_SUFFIXES: tuple = ()
    _OUTPUT_FORT_FILES: tuple = ()

    def __new__(cls, *args, **kwargs):
        if cls is OASES:
            raise ConfigurationError(
                "OASES is the abstract base for OAST/OASN/OASR/OASP. "
                "Instantiate a sub-model directly, or use "
                "OASES.for_mode(run_mode=...) to pick one by RunMode."
            )
        return super().__new__(cls)

    @classmethod
    def for_mode(cls, run_mode: RunMode = RunMode.COHERENT_TL, *,
                 broadband: bool = False, **kwargs) -> PropagationModel:
        """Instantiate the OASES sub-model that handles ``run_mode``.

        ``COHERENT_TL`` → ``OAST`` (``OASP`` when ``broadband=True``);
        ``COVARIANCE``/``REPLICA`` → ``OASN``; ``REFLECTION`` → ``OASR``;
        ``BROADBAND``/``TIME_SERIES`` → ``OASP``. ``**kwargs`` forward to the
        chosen sub-model (a kwarg it doesn't accept raises ``TypeError``).
        """
        dispatch = {
            RunMode.COHERENT_TL: OASP if broadband else OAST,
            RunMode.COVARIANCE: OASN,
            RunMode.REPLICA: OASN,
            RunMode.REFLECTION: OASR,
            RunMode.BROADBAND: OASP,
            RunMode.TIME_SERIES: OASP,
        }
        target = dispatch.get(run_mode)
        if target is None:
            raise UnsupportedFeatureError(
                'OASES', str(run_mode),
                alternatives=[str(m) for m in dispatch],
                alternatives_label='run modes',
            )
        return target(**kwargs)

    def _max_receiver_depth(self, env: Environment) -> float:
        """Deepest depth OASES resolves the field at: the full media stack.

        OASES meshes the sediment layers as media in their own right, so the
        field is defined through them and a source or receiver inside the
        seabed is a supported geometry — buried and sub-bottom sources are a
        large part of what the seismo-acoustic family is for.
        """
        return self._total_media_depth(env)

    def _require_output(self, candidates, work_dir: Path, base_name: str, *,
                        what: str, process=None, hint: str = '') -> Path:
        """First of ``candidates`` that exists and is non-empty.

        None of them means the binary failed. OASES writes no print file and
        has no equivalent of the Acoustics-Toolbox ``FatalError`` module: its
        diagnostics go to stdout (``WRITE(6,*)``) and to gfortran's stop-code
        echo on stderr, both of which exit 0. ``process`` is the binary's
        :class:`subprocess.CompletedProcess`; its streams are carried into the
        raised :class:`ModelExecutionError` because nothing else holds them.
        """
        for candidate in candidates:
            path = Path(candidate)
            if path.exists() and path.stat().st_size > 0:
                return path
        checked = ', '.join(str(c) for c in candidates)
        raise ModelExecutionError(
            self.model_name,
            return_code=getattr(process, 'returncode', 0),
            stdout=_tail(getattr(process, 'stdout', None)),
            stderr=(
                f"{self.model_name} did not produce {what}. Checked: "
                f"{checked}." + (f" {hint}" if hint else "")
                + _stream_block('binary stderr',
                                getattr(process, 'stderr', None))
            ),
        )

    def _execute(self, base_name: str, work_dir: Path):
        """Run the OASES binary and return its ``CompletedProcess``.

        FOR005 stays as stdin per OASES docs. The completed process is
        returned rather than dropped: OASES writes no print file, so its
        stdout/stderr are the only record of what it did.
        """
        env = _oases_subprocess_env(base_name, **self._FOR_FILES)
        for name in self._OUTPUT_FORT_FILES:
            Path(work_dir).joinpath(name).unlink(missing_ok=True)
        return self._run_and_attach_prt(
            [str(self._exe)], work_dir, base_name, env=env,
            stale_outputs=self._OUTPUT_SUFFIXES,
        )


class OAST(OASES):
    """
    OAST - OASES Transmission Loss Model

    Computes transmission loss using wavenumber integration. Best for
    range-independent environments with depth-dependent sound speed profiles.

    Parameters
    ----------
    executable : Path, optional
        Path to OAST binary. Auto-detected if ``None``.
    compute_contour : bool, optional
        Add ``'C'`` option (range-depth contour plot). Effective default
        ``False``.
    compute_depth_average : bool, optional
        Add ``'A'`` option (depth-averaged TL). Effective default ``False``.
    complex_contour : bool, optional
        ``'J'`` option (complex integration contour). Effective default
        ``True``.
    options : str, optional
        Raw OASES options string (e.g. ``'N J T C'``), written verbatim;
        ``None`` derives it from ``compute_contour`` /
        ``compute_depth_average`` / ``complex_contour``. Combining a raw
        string with any of those three flags raises
        ``ConfigurationError`` — the string replaces the whole option
        line, so a flag passed alongside it would be discarded.
    integration_offset : float
        Wavenumber-integration contour offset (dB/wavelength). Default 0.
    nw_samples : int
        Number of wavenumber samples; ``-1`` lets OASES choose.
    plot_rmin, plot_rmax : float, optional
        TL plot range axis bounds (m); ``None`` → 0 /
        ``receiver.range_max``.
    vrec : float
        Receiver velocity (m/s) for OAST's Doppler compensation, the
        fifth token of the frequency line. Read only when the option
        string carries lowercase ``'d'`` (``unoast31.f:125-127``);
        default 0.
    dip_angle : float, optional
        Fault dip angle (degrees) for the dip-slip moment source that
        option ``'4'`` selects (``unoast31.f:1117-1122``). INSRC reads it
        from the source record — as the second token without ``'L'``
        (``oaseun31.f:1114``), as the seventh with it
        (``oaseun31.f:1089``). ``None`` writes 0 when ``'4'`` is present
        and raises when it is not.
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Range-independent wavenumber integration; consumes layered seabed
    columns natively. **Collapse defaults (overrides of
    :data:`DEFAULT_COLLAPSE`).** Per-model: ``'ssp': 'mean'``,
    ``'bottom_range': 'median'`` (the layer stack is kept).

    ``COHERENT_TL`` here returns ``Field.kind == 'tl'`` (real dB), not the
    ``'pressure'`` the other OASES sub-model returns for the same run mode:
    OAST's ``.plt`` carries only real TL, so there is no complex pressure to
    hand back and no phase reference to tag. Use :class:`OASP` when the
    consumer needs complex pressure — coherent array processing, phase, or
    off-grid receiver ranges through sharp interference nulls.

    Examples
    --------
    >>> from uacpy.models import OAST
    >>> oast = OAST()
    >>> result = oast.run(env, source, receiver)
    """

    # Declarative metadata (see PropagationModel / ModelSpec). OAST:
    # range-independent wavenumber integration; multi-layer fluid + elastic
    # bottom honoured. Single spectral solve → mean SSP / median bottom column.
    spec = ModelSpec(
        modes=(RunMode.COHERENT_TL,),
        supports={'layered_bottom', 'elastic_media',
                  'rough_surface', 'rough_bottom'},
        collapse={'ssp': 'mean', 'bottom_range': 'median'},
    )
    source = 'oases'

    def __init__(
        self,
        executable: Optional[Path] = None,
        compute_contour: Optional[bool] = None,
        compute_depth_average: Optional[bool] = None,
        complex_contour: Optional[bool] = None,
        options: Optional[str] = None,
        integration_offset: float = 0.0,
        nw_samples: int = -1,
        plot_rmin: Optional[float] = None,
        plot_rmax: Optional[float] = None,
        vrec: float = 0.0,
        dip_angle: Optional[float] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: float = 600.0,
        collapse: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            cleanup=cleanup, timeout=timeout, collapse=collapse,
        )
        # Kept as passed so ``copy()`` round-trips and so an explicit flag
        # alongside a raw ``options`` string is distinguishable from the
        # default. Resolved by ``_resolve_options``.
        self.compute_contour = compute_contour
        self.compute_depth_average = compute_depth_average
        self.complex_contour = complex_contour
        # Raw OASES option string (e.g. ``'N J T C'``). ``None`` lets the
        # wrapper derive it from compute_contour / compute_depth_average /
        # complex_contour; a raw string replaces the deck's option line
        # outright, so the two ways of specifying it are exclusive.
        self.options = options
        if options is not None:
            pinned = [n for n in ('compute_contour', 'compute_depth_average',
                                  'complex_contour')
                      if getattr(self, n) is not None]
            if pinned:
                raise ConfigurationError(
                    f"OAST: options={options!r} replaces the whole option "
                    f"line, so {', '.join(pinned)} would be discarded "
                    f"silently. Pass either the raw string or the typed "
                    f"flags, not both — note the derived string includes "
                    f"'J' (complex integration contour) by default, which a "
                    f"raw string must repeat to keep."
                )
        self.integration_offset = float(integration_offset)
        # ``-1`` lets the OASES kernel pick its own wavenumber sample count.
        self.nw_samples = int(nw_samples)
        # Plot-axis bounds in metres. ``None`` → 0 / receiver.range_max.
        self.plot_rmin = float(plot_rmin) if plot_rmin is not None else None
        self.plot_rmax = float(plot_rmax) if plot_rmax is not None else None
        # VREC, the fifth frequency-line token OAST reads only under the
        # lowercase 'd' (Doppler) option.
        self.vrec = float(vrec)
        # DA, the dip angle INSRC reads only under the '4' (dip-slip) option.
        self.dip_angle = float(dip_angle) if dip_angle is not None else None

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        #
        # Keep the user's ``executable`` arg verbatim (``None`` when
        # auto-detected) so ``model.copy()`` re-resolves the binary instead of
        # re-pinning the already-resolved absolute path. The resolved path
        # lives in ``self._exe``.
        self.executable = Path(executable) if executable is not None else None
        if self.executable is None:
            self._exe = _oases_find_executable(self, 'oast')
        else:
            self._exe = self.executable

        if not self._exe.exists():
            raise ExecutableNotFoundError('OAST', str(self._exe))

    def _resolve_options(self) -> str:
        """The OASES option line for this run.

        A raw ``options`` string is written verbatim; otherwise the letters
        are derived from the typed flags on top of ``'N T'`` (normal stress
        + TL table). The constructor rejects the two being combined.
        """
        if self.options is not None:
            return self.options
        opt = ['N', 'T']
        if self.complex_contour is None or self.complex_contour:
            opt.append('J')
        if self.compute_contour:
            opt.append('C')
        if self.compute_depth_average:
            opt.append('A')
        return ' '.join(opt)

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        *,
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        output_duration: Optional[float] = None,
    ) -> Result:
        """
        Run OAST transmission loss computation

        Parameters
        ----------
        env : Environment
            Ocean environment (must be range-independent)
        source : Source
            Acoustic source
        receiver : Receiver
            Receiver array

        Returns
        -------
        result : Result
            Transmission loss field
        """
        self._reject_unsupported_run_kwargs(
            frequencies=frequencies, source_waveform=source_waveform,
            sample_rate=sample_rate, output_duration=output_duration)
        run_mode = self._resolve_run_mode(run_mode)

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)
        fm = self._setup_file_manager()

        try:
            base_name = 'oast_run'
            input_file = fm.get_path(f'{base_name}.dat')

            user_options = self._resolve_options()

            self._log(f"Writing OAST input file: {input_file} (options={user_options})")
            writer_kwargs: dict = {
                'integration_offset': self.integration_offset,
                'nw_samples': self.nw_samples,
                'vrec': self.vrec,
            }
            if self.plot_rmin is not None:
                writer_kwargs['plot_rmin'] = self.plot_rmin
            if self.plot_rmax is not None:
                writer_kwargs['plot_rmax'] = self.plot_rmax
            if self.dip_angle is not None:
                writer_kwargs['dip_angle'] = self.dip_angle
            write_oast_input(
                filepath=input_file,
                env=env,
                source=source,
                receiver=receiver,
                options=user_options,
                **writer_kwargs,
            )

            proc = self._execute(base_name, fm.work_dir)

            # Read output - OAST creates .plt (data) and .plp (metadata) files
            # Per OASES documentation: FOR019 -> .plp, FOR020 -> .plt
            output_file = self._require_output(
                [fm.get_path(f'{base_name}.plt')], fm.work_dir, base_name,
                what='a TL table (FOR020 .plt)', process=proc,
                hint=('Compare against the reference decks in '
                      'third_party/oases/tloss/.'),
            )

            self._log(f"Reading OAST output: {output_file}")
            tl_data, native_depths, native_ranges, metadata = read_oast_tl(
                filepath=output_file,
                receiver_depths=receiver.depths,
            )

            kw = self._result_kwargs(
                source,
                backend='oast',
                frequencies=float(np.atleast_1d(source.frequencies)[0]),
            )
            kw['metadata'].update(metadata)
            native = Field(
                data=tl_data,
                coords={'depth': native_depths, 'range': native_ranges},
                **kw,
            )
            receiver_ranges = np.atleast_1d(np.asarray(receiver.ranges, dtype=float))
            receiver_depths = np.atleast_1d(np.asarray(receiver.depths, dtype=float))
            ranges_match = (
                len(native_ranges) == len(receiver_ranges)
                and np.allclose(native_ranges, receiver_ranges)
            )
            if ranges_match:
                result = native
            else:
                # OAST writes only real TL (dB) to .plt — there is no complex
                # pressure on disk — so off-grid receiver ranges are obtained
                # by linear interpolation IN dB. That over-fills sharp
                # interference minima (a true −80 dB null between −40 dB
                # neighbours reads back ≈ −50 dB). Align receiver.ranges to the
                # OAST native FFT grid (ranges_match) to avoid the smoothing,
                # or use OASP (.trf complex pressure) when null depth matters.
                warnings.warn(
                    "OAST: receiver.ranges do not match the OAST native FFT "
                    "range grid, so TL is linearly interpolated IN dB. This "
                    "smears sharp interference minima (a true −80 dB null "
                    "between −40 dB neighbours reads back ≈ −50 dB). Align "
                    "receiver.ranges to the native grid, or use OASP (.trf "
                    "complex pressure) when null depth matters.",
                    UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
                )
                result = native.resample_to(
                    ranges=receiver_ranges, depths=receiver_depths)
                result.metadata['oast_native_ranges'] = native_ranges
                result.metadata['interpolated'] = True

            self._attach_output_paths(
                result, fm.work_dir, base_name,
                primary_files=(('plt_file', '.plt'),),
            )

            self._log("OAST simulation complete")
            return result

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    # Mirrors third_party/oases/bin/oast. FOR002 is the external source
    # array option 'l' reads; FOR023 the tabulated reflection coefficient
    # options 't'/'b' read (oaseun31.f:3727, :3757). uacpy rejects all three
    # letters (_UNWRITTEN_OPTION_BLOCKS), so both units stay unopened.
    _FOR_FILES = {
        'FOR002': 'src',
        'FOR023': 'trc',
    }
    _OUTPUT_SUFFIXES = ('.plt',)


class OASN(OASES):
    """
    OASN — OASES Noise, Covariance Matrices and Signal Replicas.

    Per the OASES manual (``third_party/oases/doc/oasn.tex``) OASN produces
    two frequency-domain array products:

    - **Covariance matrices** (option ``N`` → ``.xsm``): hydrophone ×
      hydrophone correlation per frequency. Used for ambient-noise
      characterisation or as an MFP measurement covariance.
    - **Replica fields** (option ``R`` → ``.rpo``): array response per
      candidate source position per frequency. Used as MFP templates.

    For depth-eigenfunction normal modes use :class:`Kraken`.

    Parameters
    ----------
    executable : Path, optional
        Path to OASN binary. Auto-detected if ``None``.
    options : str, optional
        Custom OASES options string; ``None`` derives it from the run
        mode (``N`` for COVARIANCE, ``R`` for REPLICA).
    surface_noise_level : float
        Surface-generated noise spectral level (dB re 1 µPa²/Hz),
        Block VI. OASES disables the source when ``abs(level) < 0.01``
        (``oasnun22.f:183``); a level at or below ``-0.01`` is not "off" —
        it names the Fortran unit number of a source-spectrum file uacpy
        does not write, so it is rejected. Requires a vacuum or air
        ``env.surface`` (``oasnun22.f:187``).
    white_noise_level : float
        Uncorrelated (white) noise spectral level per hydrophone
        (dB re 1 µPa²/Hz). 0 disables.
    deep_noise_level : float
        Deep broad-area source spectral level (dB re 1 µPa²/Hz). OASES
        uses two thresholds that disagree at exactly 0.01: the source
        radiates only for ``level > 0.01`` (``oasnun22.f:233`` skips it on
        ``.LE.0.01``) while Block VIII is read for ``level >= 0.01``
        (``oasnun22.f:324``). Unlike the surface level, a negative value
        simply disables it — the deep source has no spectrum-file form.
    deep_source_depth : float, optional
        Depth (m) of the deep broad-area noise source sheet; ``None``
        → half the water depth. Only written when ``deep_noise_level``
        is non-zero.
    discrete_sources : list of dict, optional
        Point sources; each dict may carry ``'depth'`` (m), ``'x'``
        (m), ``'y'`` (m) and ``'level'`` (dB, non-negative) — the four
        fields OASES reads (``oasnun22.f:380``). Any other key raises
        ``ConfigurationError``; OASES has no per-source phase.
    xmin, xmax : float, optional
        Replica candidate-grid x bounds (m); ``None`` → OASES defaults
        (100 / 10000).
    nx : int
        Number of replica grid points in x. Default 50.
    ymin, ymax : float, optional
        Replica candidate-grid y bounds (m); ``None`` → 0 / 0.
    ny : int
        Number of replica grid points in y. Default 1.
    zmin, zmax : float, optional
        Replica candidate-grid depth bounds (m); ``None`` → 10 /
        ``env.depth - 10``.
    nz : int
        Number of replica grid points in depth. Default 20.
    c_low, c_high : float, optional
        Phase-speed bounds (m/s) for the wavenumber integrations,
        applied to both the noise and replica blocks; ``None`` →
        ``0.95 · min(c_water)`` and ``1e8``.
    integration_offset : float
        Wavenumber-integration contour offset (dB/wavelength). Default 0.
    nw_samples : int
        Number of wavenumber samples; ``-1`` lets OASES choose.
    plot_rmin, plot_rmax : float, optional
        TL plot range axis bounds (m).
    vrec : float
        Vertical receiver velocity (m/s) for Doppler. Default 0.
    offdb : float, optional
        Single-mode horizontal offset (dB).
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Supported run modes: ``RunMode.COVARIANCE`` (``.xsm`` →
    :class:`Covariance`) and ``RunMode.REPLICA`` (``.rpo`` →
    :class:`Replicas`). The convenience methods
    ``compute_covariance(...)`` / ``compute_replicas(...)`` route to
    these.

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    Per-model: ``'ssp': 'mean'``, ``'bottom_range': 'median'`` (the
    layer stack is kept).

    Examples
    --------
    >>> from uacpy.models import OASN
    >>> oasn = OASN()
    >>> cov = oasn.compute_covariance(env, source, receiver)
    >>> # cov.covariance has shape (n_frequencies, n_rcv, n_rcv)
    """

    # Declarative metadata (see PropagationModel / ModelSpec). OASN:
    # range-independent covariance / replica field; multi-layer bottom
    # honoured. Single spectral solve per frequency → mean SSP / median column.
    spec = ModelSpec(
        modes=(RunMode.COVARIANCE, RunMode.REPLICA),
        supports={'layered_bottom', 'elastic_media',
                  'rough_surface', 'rough_bottom'},
        collapse={'ssp': 'mean', 'bottom_range': 'median'},
    )
    source = 'oases'

    def __init__(
        self,
        executable: Optional[Path] = None,
        # Output / option control
        options: Optional[str] = None,
        # Noise field (Block VI): broad-area sources expressed as
        # spectral levels (dB re 1 µPa²/Hz) at three depths.
        surface_noise_level: float = 0.0,
        white_noise_level: float = 0.0,
        deep_noise_level: float = 0.0,
        deep_source_depth: Optional[float] = None,
        # Discrete (point) sources — list of dicts; each may carry
        # 'depth' (m), 'x' (m), 'y' (m), 'level' (dB).
        discrete_sources: Optional[list] = None,
        # Replica candidate-position grid (Block X). x/y/z in metres on
        # the public API, converted to km at write time. ``None`` lets
        # the writer apply OASES defaults (10 / depth-10 in z, 100 /
        # 10000 in x, 0 / 0 in y).
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        nx: int = 50,
        ymin: Optional[float] = None,
        ymax: Optional[float] = None,
        ny: int = 1,
        zmin: Optional[float] = None,
        zmax: Optional[float] = None,
        nz: int = 20,
        # Phase-speed bounds for the wavenumber integrations (m/s).
        # Applied identically to OASN Block VIII (noise / discrete
        # sources) and Block X (replica generator). ``None`` → writer
        # derives c_water_min * 0.95 (c_low) and 1e8 (c_high).
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        # Wavenumber-axis sampling & TL plot axes.
        integration_offset: float = 0.0,
        nw_samples: int = -1,
        plot_rmin: Optional[float] = None,
        plot_rmax: Optional[float] = None,
        # Frequency-line extras: vrec (m/s, vertical receiver velocity
        # for Doppler), offdb (single-mode horizontal offset).
        vrec: float = 0.0,
        offdb: Optional[float] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: float = 600.0,
        collapse: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            cleanup=cleanup, timeout=timeout, collapse=collapse,
        )
        self.options = options
        self.surface_noise_level = float(surface_noise_level)
        self.white_noise_level = float(white_noise_level)
        self.deep_noise_level = float(deep_noise_level)
        self.deep_source_depth = (
            float(deep_source_depth) if deep_source_depth is not None else None
        )
        self.discrete_sources = list(discrete_sources) if discrete_sources else None
        self.xmin = xmin
        self.xmax = xmax
        self.nx = int(nx)
        self.ymin = ymin
        self.ymax = ymax
        self.ny = int(ny)
        self.zmin = zmin
        self.zmax = zmax
        self.nz = int(nz)
        self.c_low = c_low
        self.c_high = c_high
        self.integration_offset = float(integration_offset)
        self.nw_samples = int(nw_samples)
        # OASN's deck has no range-axis or receiver-velocity field: it emits
        # covariance matrices and replicas, not TL-vs-range, and VREC appears
        # nowhere in oasnun22.f / unoasn22.f (it is OAST's Doppler-only slot).
        # Accepting these silently would advertise knobs that cannot reach the
        # binary, so they are rejected at construction instead.
        for name, value in (('plot_rmin', plot_rmin), ('plot_rmax', plot_rmax)):
            if value is not None:
                raise ConfigurationError(
                    f"OASN({name}=...) has no effect: OASN writes covariance "
                    f"matrices and signal replicas, not a range-axis plot "
                    f"block.",
                    remediation=("Use OAST for transmission loss over a range "
                                 "axis, or drop the argument."),
                )
        if vrec:
            raise ConfigurationError(
                "OASN(vrec=...) has no effect: VREC is OAST's Doppler "
                "receiver-velocity field (the 5-token frequency line enabled "
                "by option 'd') and does not exist in OASN's input format.",
                remediation="Use OAST with the 'd' option, or drop vrec.",
            )
        self.plot_rmin = None
        self.plot_rmax = None
        self.vrec = 0.0
        # Same OASES field as integration_offset (unoasn22.f:142 reads
        # FREQ1 FREQ2 NFREQ OFFDBIN); an explicit value wins in the writer.
        self.offdb = float(offdb) if offdb is not None else None

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        #
        # Keep the user's ``executable`` arg verbatim (``None`` when
        # auto-detected) so ``model.copy()`` re-resolves the binary instead of
        # re-pinning the already-resolved absolute path. The resolved path
        # lives in ``self._exe``.
        self.executable = Path(executable) if executable is not None else None
        if self.executable is None:
            self._exe = _oases_find_executable(self, 'oasn2_bin')
        else:
            self._exe = self.executable

        if not self._exe.exists():
            raise ExecutableNotFoundError('OASN', str(self._exe))

    def _has_noise_field(self) -> bool:
        """True when any Block VI source was configured on this instance."""
        return bool(self.surface_noise_level or self.white_noise_level
                    or self.deep_noise_level or self.discrete_sources)

    def _build_writer_kwargs(self) -> dict:
        """Materialise the writer's expected kwarg dict from ``self.*``.

        Translates uacpy's public API (singular names, metres) to the
        OASES file format the writer consumes (plural names for
        cmins/cmaxs, km for x/y axes). Only keys whose stored value is
        non-default are emitted, so the writer falls back to its own
        defaults for everything the user did not set.
        """
        kw: dict = {
            'integration_offset': self.integration_offset,
            'nw_samples': self.nw_samples,
            'surface_noise_level': self.surface_noise_level,
            'white_noise_level': self.white_noise_level,
            'deep_noise_level': self.deep_noise_level,
        }
        if self.deep_source_depth is not None:
            kw['deep_source_depth'] = self.deep_source_depth
        if self.offdb is not None:
            kw['offdb'] = self.offdb

        # Phase-speed bounds applied identically to OASN's noise and
        # replica integrations (singular on the public API, plural-with-
        # suffix in the writer per OASES's CMINS/CMAXS variable names).
        if self.c_low is not None:
            # Under its own name for the surface/deep *noise* blocks, and
            # under the writer's CMINS-style names for the discrete-source and
            # replica integrations — the docstring promises all four.
            kw['c_low'] = self.c_low
            kw['cmins_discrete'] = self.c_low
            kw['cmins_replica'] = self.c_low
        if self.c_high is not None:
            kw['c_high'] = self.c_high
            kw['cmaxs_discrete'] = self.c_high
            kw['cmaxs_replica'] = self.c_high

        # Replica grid x/y in km on disk; z in metres. nx/ny/nz always
        # forwarded so the writer reuses uacpy defaults rather than its
        # own OASES-doc ones when the user did not override.
        kw['replica_nx'] = self.nx
        kw['replica_ny'] = self.ny
        kw['replica_nz'] = self.nz
        if self.xmin is not None:
            kw['replica_xmin'] = float(m_to_km(self.xmin))
        if self.xmax is not None:
            kw['replica_xmax'] = float(m_to_km(self.xmax))
        if self.ymin is not None:
            kw['replica_ymin'] = float(m_to_km(self.ymin))
        if self.ymax is not None:
            kw['replica_ymax'] = float(m_to_km(self.ymax))
        if self.zmin is not None:
            kw['replica_zmin'] = self.zmin
        if self.zmax is not None:
            kw['replica_zmax'] = self.zmax

        # ``discrete_sources`` carry per-source x/y in metres (uacpy
        # API); OASES writes them in km (oasn.tex:343).
        if self.discrete_sources:
            converted = []
            for ds in self.discrete_sources:
                ds_km = dict(ds)
                if 'x' in ds_km:
                    ds_km['x'] = float(m_to_km(ds_km['x']))
                if 'y' in ds_km:
                    ds_km['y'] = float(m_to_km(ds_km['y']))
                converted.append(ds_km)
            kw['discrete_sources'] = converted

        return kw

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        *,
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        output_duration: Optional[float] = None,
    ) -> Result:
        """
        Run OASN.

        Parameters
        ----------
        run_mode : RunMode, optional
            ``RunMode.COVARIANCE`` (default) → :class:`Covariance`.
            ``RunMode.REPLICA`` → :class:`Replicas` (build the OASN
            instance with ``xmin``/``xmax``/``zmin``/``zmax``/… already
            set on the constructor; sweeps go through ``model.copy(...)``).

        Returns
        -------
        Covariance or Replicas
        """
        self._reject_unsupported_run_kwargs(
            frequencies=frequencies, source_waveform=source_waveform,
            sample_rate=sample_rate, output_duration=output_duration)
        run_mode = self._resolve_run_mode(run_mode)

        # OASN expresses the frequency axis as (fmin, fmax, N) like OASR
        # and OASP, so a non-equispaced source.frequencies vector would
        # be silently regridded by _resolve_freq_sweep inside the writer.
        # Warn (and resample) here so the user sees what is happening.
        source_freqs = np.atleast_1d(np.asarray(source.frequencies, dtype=float))
        if source_freqs.size > 1:
            _oases_resample_frequencies(source_freqs, self.model_name)

        # The OASN writer places a vertical array at x = y = 0;
        # receiver.ranges never reaches the deck.
        rcv_ranges = np.atleast_1d(np.asarray(receiver.ranges, dtype=float))
        if rcv_ranges.size > 1 or (rcv_ranges.size == 1 and rcv_ranges[0] > 0.0):
            warnings.warn(
                "OASN: receiver.ranges is ignored — OASN models a vertical "
                "array at x = y = 0 (depths only). Use the replica grid "
                "(xmin/xmax/...) for horizontal apertures.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)

        writer_kwargs = self._build_writer_kwargs()

        # The OASN options block must include the file output the run
        # mode produces. The user-supplied options (or default 'J') win
        # for additional letters (F, …); N (covariance) or R (replica)
        # is added automatically based on run_mode.
        user_options = self.options or 'J'
        opt_tokens = set(user_options.split())
        if run_mode == RunMode.COVARIANCE:
            opt_tokens.add('N')
        else:
            opt_tokens.add('R')
        # NOIPAR — and with it every noise argument below — runs only under
        # `IF (CALNSE.or.trfout)` (unoasn22.f:173-174), and CALNSE comes from
        # 'N'. A replica run that was given a noise field therefore needs 'N'
        # too, or the levels it was handed would never reach the deck.
        if self._has_noise_field():
            opt_tokens.add('N')
        options = ' '.join(sorted(opt_tokens))

        fm = self._setup_file_manager()
        try:
            base_name = 'oasn_run'
            input_file = fm.get_path(f'{base_name}.dat')
            self._log(f"Writing OASN input file: {input_file} (options={options})")
            write_oasn_input(
                filepath=input_file,
                env=env,
                source=source,
                receiver=receiver,
                options=options,
                **writer_kwargs,
            )
            proc = self._execute(base_name, fm.work_dir)

            if run_mode == RunMode.COVARIANCE:
                cov_path = self._require_output(
                    [fm.get_path(f'{base_name}.xsm'), fm.get_path('fort.16')],
                    fm.work_dir, base_name, what='a covariance file',
                    process=proc,
                )
                self._log(f"Reading OASN covariance file: {cov_path}")
                cov_data = read_oasn_covariance(cov_path)

                rcv_pos = None
                rcv_depths = np.atleast_1d(np.asarray(receiver.depths, dtype=float))
                if rcv_depths.size == cov_data['covariance'].shape[1]:
                    # OASES OASN config places receivers along the array;
                    # only depth varies in the typical uacpy usage.
                    rcv_pos = np.zeros((rcv_depths.size, 3), dtype=float)
                    rcv_pos[:, 2] = rcv_depths

                freqs = _oasn_freq_axis(cov_data)
                kw = self._result_kwargs(
                    source,
                    backend='oasn',
                    frequencies=freqs,
                    n_receivers=cov_data['n_receivers'],
                    title=cov_data.get('title', ''),
                )
                cov_result = Covariance(
                    covariance=cov_data['covariance'],
                    receiver_positions=rcv_pos,
                    **kw,
                )
                self._attach_output_paths(
                    cov_result, fm.work_dir, base_name,
                    primary_files=(('xsm_file', '.xsm'),),
                )
                return cov_result

            # RunMode.REPLICA
            rep_path = self._require_output(
                [fm.get_path(f'{base_name}.rpo'), fm.get_path('fort.14')],
                fm.work_dir, base_name, what='a replica file', process=proc,
                hint=('Set the replica grid via OASN(xmin=…, xmax=…, nx=…, '
                      'zmin=…, zmax=…, nz=…) on the constructor.'),
            )
            self._log(f"Reading OASN replica file: {rep_path}")
            rep_data = read_oasn_replicas(rep_path)
            # x/y come back in km from the .rpo file; expose metres.
            replica_z = np.linspace(rep_data['z_min'], rep_data['z_max'], rep_data['n_z'])
            replica_x = np.linspace(rep_data['x_min'], rep_data['x_max'], rep_data['n_x']) * 1000.0
            replica_y = np.linspace(rep_data['y_min'], rep_data['y_max'], rep_data['n_y']) * 1000.0
            freqs = _oasn_freq_axis(rep_data)
            kw = self._result_kwargs(
                source,
                backend='oasn',
                frequencies=freqs,
                n_receivers=rep_data['n_receivers'],
                title=rep_data.get('title', ''),
            )
            rep_result = Replicas(
                replicas=rep_data['replicas'],
                replica_z=replica_z,
                replica_x=replica_x,
                replica_y=replica_y,
                receiver_positions=rep_data.get('receiver_positions'),
                **kw,
            )
            self._attach_output_paths(
                rep_result, fm.work_dir, base_name,
                primary_files=(('rpo_file', '.rpo'),),
            )
            return rep_result

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    # ``compute_covariance`` / ``compute_replicas`` come from the base class
    # (RunMode.COVARIANCE / RunMode.REPLICA dispatch).

    # Mirrors third_party/oases/bin/oasn: replica vectors on unit 14,
    # covariance on unit 16 (oasmun21_bin.f:335 reads FOR016 back), and the
    # unit-26 echo of the array and noise levels INPRCV opens at
    # oasnun22.f:37.
    _FOR_FILES = {
        'FOR014': 'rpo',
        'FOR016': 'xsm',
        'FOR026': 'chk',
    }
    _OUTPUT_SUFFIXES = ('.xsm', '.rpo', '.plt')
    _OUTPUT_FORT_FILES = ('fort.14', 'fort.16')


class OASR(OASES):
    """
    OASR - OASES Reflection Coefficients Model

    Computes plane wave reflection coefficients at the bottom interface.

    Parameters
    ----------
    executable : Path, optional
        Path to OASR binary. Auto-detected if ``None``.
    angles : ndarray, optional
        Uniformly spaced angle grid (deg). Default
        ``linspace(0, 90, 181)``.
    angle_type : str, optional
        ``'grazing'`` (OASES native, default) | ``'incidence'``
        (converted via ``grazing = 90 - incidence``).
    reflection_type : str, optional
        ``'P-P'`` (effective default) | ``'P-SV'`` | ``'P-Slow'`` (Biot
        only) | ``'transmission'``. Translates to OASR option letter
        (``'N'`` / ``'S'`` / ``'B'`` / ``'t'``). Mutually exclusive with
        a raw ``options`` string.
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Range-independent reflection-vs-angle/freq solver. Supports
    elastic / poro-elastic bottom layers. Returns reflection magnitude
    and phase per ``(frequency, angle)``. Pass ``freq_min``,
    ``freq_max``, ``n_frequencies`` on ``run()`` for a broadband sweep.

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    Per-model: ``'bottom_range': 'median'`` (the layer stack is
    kept). SSP collapse left at the global ``'r0'`` because the
    SSP boundary speed is essentially irrelevant to the reflection
    coefficient.

    Examples
    --------
    >>> from uacpy.models import OASR
    >>> oasr = OASR(angles=np.linspace(0, 90, 100))
    >>> refl = oasr.run(env, source, receiver)
    """

    # Declarative metadata (see PropagationModel / ModelSpec). OASR:
    # range-independent reflection vs angle/freq; layered bottom honoured.
    # Only the bottom stack matters — SSP collapse left at the global default
    # ('r0'), as the SSP is essentially irrelevant to the reflection coeff.
    spec = ModelSpec(
        modes=(RunMode.REFLECTION,),
        supports={'layered_bottom', 'elastic_media',
                  'rough_surface', 'rough_bottom'},
        collapse={'bottom_range': 'median'},
    )
    source = 'oases'

    def __init__(
        self,
        executable: Optional[Path] = None,
        angles: Optional[np.ndarray] = None,
        angle_type: str = 'grazing',
        reflection_type: Optional[str] = None,
        options: Optional[str] = None,
        angle_output_increment: Optional[int] = None,
        freq_output_increment: Optional[int] = None,
        interface_roughness: Optional[list] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: float = 600.0,
        collapse: Optional[Dict[str, str]] = None,
    ):
        """
        Parameters
        ----------
        executable : Path, optional
            Path to OASR executable. Auto-detected if None.
        angles : ndarray, optional
            Angle grid in degrees, uniformly spaced. Default:
            ``np.linspace(0, 90, 181)``. OASR's deck carries only
            ``(ANGLE1, ANGLE2, NANG)`` and generates the grid itself
            (``unoasr21.f:173-177``), so a non-uniform array raises
            ``ConfigurationError`` rather than being silently resampled.
        angle_type : str, optional
            'grazing' (OASES native) or 'incidence' (converted via
            ``grazing = 90 - incidence`` before being written to the input
            file). Default: 'grazing'.
        reflection_type : str, optional
            One of 'P-P' (effective default), 'P-SV', 'P-Slow' (Biot
            only), or 'transmission'. Translates to the OASR option
            letter ('N' / 'S' / 'B' / 't').
        options : str, optional
            Raw OASES option string, written verbatim. ``None`` lets the
            wrapper derive it from ``reflection_type``. Passing both
            raises ``ConfigurationError`` — the deck would run whichever
            reflection the raw string selects, not the named one.
        angle_output_increment : int, optional
            Decimation factor (OASR's ``NAOU``) for the output angle
            table. ``None`` keeps every sample.
        freq_output_increment : int, optional
            Decimation factor (OASR's ``NFOU``, ``unoasr21.f:116``) for
            the reflection-vs-frequency output. ``None`` → ``n_freq //
            10``.
        interface_roughness : list, optional
            Per-interface RMS roughness in metres, one entry per layer
            record, ordered top → bottom. Entry 0 is OASR's layer-1
            (water half-space) record, whose RG INENVI discards
            (``oaseun31.f:377``); entry 1 is the reflecting water/seabed
            interface. An entry may be a float, or a ``(RG, CL, M)``
            tuple / ``{'RG', 'CL', 'M'}`` dict to select the Goff-Jordan
            roughness spectrum. Unset entries fall back to the
            environment's own interface roughness.
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            cleanup=cleanup, timeout=timeout, collapse=collapse,
        )
        self.angles = (
            np.asarray(angles, dtype=float) if angles is not None else None
        )
        self.angle_type = angle_type
        # Kept as passed so ``copy()`` round-trips and so an explicit value
        # alongside a raw ``options`` string is distinguishable from the
        # default. Resolved by ``_resolve_reflection_type``.
        self.reflection_type = reflection_type
        self.options = options
        if options is not None and reflection_type is not None:
            raise ConfigurationError(
                f"OASR: options={options!r} replaces the whole option line, "
                f"so reflection_type={reflection_type!r} would be recorded "
                f"in the result metadata without ever reaching the deck. "
                f"Pass either the raw string or reflection_type, not both."
            )
        self.angle_output_increment = (
            int(angle_output_increment)
            if angle_output_increment is not None else None
        )
        self.freq_output_increment = (
            int(freq_output_increment)
            if freq_output_increment is not None else None
        )
        self.interface_roughness = (
            list(interface_roughness) if interface_roughness else None
        )

        # OASR is strictly a boundary-reflection solver; it does not produce
        # transmission loss. Declare that explicitly.
        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        #
        # Keep the user's ``executable`` arg verbatim (``None`` when
        # auto-detected) so ``model.copy()`` re-resolves the binary instead of
        # re-pinning the already-resolved absolute path. The resolved path
        # lives in ``self._exe``.
        self.executable = Path(executable) if executable is not None else None
        if self.executable is None:
            self._exe = _oases_find_executable(self, 'oasr')
        else:
            self._exe = self.executable

        if not self._exe.exists():
            raise ExecutableNotFoundError('OASR', str(self._exe))

    def _resolve_reflection_type(self) -> str:
        """The reflection the deck actually computes.

        Derived from the raw ``options`` letters when one is pinned, so
        the recorded provenance describes the run rather than an unused
        constructor argument. Falls back to ``'P-P'``, OASES' own default
        when no reflection letter appears.
        """
        if self.options is None:
            return self.reflection_type or 'P-P'
        from uacpy.io.oases_writer import _REFL_TYPE_TO_OPTION
        chars = set(str(self.options)) - set(' \t\n')
        named = [name for name, letter in _REFL_TYPE_TO_OPTION.items()
                 if letter in chars]
        return named[0] if len(named) == 1 else 'P-P'

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        *,
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        output_duration: Optional[float] = None,
    ) -> Result:
        """Run OASR.

        Returns a :class:`ReflectionCoefficient` with ``theta`` (1-D angle
        grid in degrees) and ``R``/``phi`` shaped ``(n_angles, n_frequencies)``
        when more than one frequency was requested. Single-frequency runs
        return 1-D ``R``/``phi``.

        OASES uses grazing angles natively. When ``angle_type='incidence'``,
        input angles are converted via ``grazing = 90 - incidence`` before
        being written to the OASR input file.

        Parameters
        ----------
        frequencies : ndarray, optional
            Explicit frequency vector (Hz). When provided, overrides
            ``source.frequencies``. OASES treats the sweep as
            equispaced — uacpy resamples and warns if your vector is not.
        source_waveform, sample_rate, output_duration
            Accepted for the polymorphic ``run()`` contract; OASR computes
            reflection coefficients only, so a non-``None`` value warns.
        """
        run_mode = self._resolve_run_mode(run_mode)
        self._warn_ignored_run_kwargs(
            run_mode,
            reason='OASR computes plane-wave reflection coefficients only',
            source_waveform=source_waveform,
            sample_rate=sample_rate,
            output_duration=output_duration,
        )

        angles = self.angles if self.angles is not None else np.linspace(0, 90, 181)

        writer_kwargs: dict = {}
        if frequencies is not None:
            freqs_arr = np.atleast_1d(np.asarray(frequencies, dtype=float))
            if freqs_arr.size == 0:
                raise ConfigurationError(
                    "OASR.run(frequencies=…) requires at least one "
                    "positive frequency."
                )
            fmin, fmax, n_freq, _ = _oases_resample_frequencies(
                freqs_arr, 'OASR',
            )
            writer_kwargs['freq_min'] = fmin
            writer_kwargs['freq_max'] = fmax
            writer_kwargs['n_frequencies'] = n_freq
        else:
            # OASR expresses the frequency axis as (fmin, fmax, N), so a
            # non-equispaced source.frequencies vector would be silently
            # regridded by the writer. Warn (and resample) here so the user
            # sees it — mirrors OASN/OASP.
            source_freqs = np.atleast_1d(
                np.asarray(source.frequencies, dtype=float))
            if source_freqs.size > 1:
                _oases_resample_frequencies(source_freqs, 'OASR')

        if self.options is not None:
            writer_kwargs['options'] = self.options
        if self.angle_output_increment is not None:
            writer_kwargs['angle_output_increment'] = self.angle_output_increment
        if self.freq_output_increment is not None:
            writer_kwargs['freq_output_increment'] = self.freq_output_increment
        if self.interface_roughness is not None:
            writer_kwargs['interface_roughness'] = self.interface_roughness

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)

        fm = self._setup_file_manager()

        try:
            base_name = 'oasr_run'
            input_file = fm.get_path(f'{base_name}.dat')

            self._log(f"Writing OASR input file: {input_file}")
            write_oasr_input(
                filepath=input_file,
                env=env,
                source=source,
                receiver=receiver,
                angles=angles,
                angle_type=self.angle_type,
                reflection_type=self._resolve_reflection_type(),
                **writer_kwargs,
            )

            proc = self._execute(base_name, fm.work_dir)

            # Under option 'T' REFLEC writes both tables on every sample —
            # slowness in s/km to unit 22 (.rco) and grazing angle in
            # degrees to unit 23 (.trc), oasjun21.f:103-104 — so the pair
            # holds the same coefficients under two abscissae and the choice
            # here is uacpy's. Option 'p' is what makes it matter: it
            # reinterprets the deck's ANGLE1/ANGLE2/NANG triple as slowness
            # (oasjun21.f:36-37 rescales ANGLE1 by 1e-3 into s/m), so under
            # 'p' the .rco abscissa is the grid the user actually asked for.
            # OASES parses the option line character by character, so 'p' can
            # be glued to other letters ('Np'); match on the character set.
            opt_chars = set(str(self.options or '')) - set(' \t\n')
            wants_slowness = 'p' in opt_chars
            search = ['.rco', '.trc'] if wants_slowness else ['.trc', '.rco']
            search += ['.023', 'fort.023']

            output_file = self._require_output(
                [fm.get_path(ext if ext.startswith('fort')
                             else f'{base_name}{ext}') for ext in search],
                fm.work_dir, base_name,
                what='a reflection-coefficient table', process=proc,
            )

            self._log(f"Reading OASR output: {output_file}")
            data = read_oasr_reflection_coefficients(output_file)

            theta_arr, R_arr, phi_arr, freqs_arr = _stack_oasr_data(data)
            field = ReflectionCoefficient(
                theta=theta_arr, R=R_arr, phi=phi_arr,
                **self._result_kwargs(
                    source,
                    backend='oasr',
                    frequencies=freqs_arr if len(freqs_arr) else None,
                    sampling_type=data.get('sampling_type', 'angle'),
                    reflection_type=self._resolve_reflection_type(),
                ),
            )
            self._attach_output_paths(
                field, fm.work_dir, base_name,
                primary_files=(
                    ('trc_file', '.trc'),
                    ('rco_file', '.rco'),
                ),
            )

            self._log("OASR simulation complete")
            return field

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    # Units 22 and 23 are the two reflection-coefficient tables option 'T'
    # opens (unoasr21.f:201-205). They are written together, not either/or:
    # oasjun21.f:103 puts slowness on 22 and :104 grazing angle on 23, which
    # is why both files can exist and run() picks by the 'p' option.
    # third_party/oases/bin/oasr names those two and no others; FOR002 and
    # FOR004 name files OASR never opens.
    _FOR_FILES = {
        'FOR002': 'src',
        'FOR004': 'trf',
        'FOR022': 'rco',
        'FOR023': 'trc',
    }
    _OUTPUT_SUFFIXES = ('.rco', '.trc', '.023', '.plt')
    _OUTPUT_FORT_FILES = ('fort.023',)


class OASP(OASES):
    """
    OASP - OASES Pulse / Broadband Transfer-Function Model

    Computes broadband acoustic transfer functions via wavenumber integration
    followed by FFT to produce time-series / pulse responses. OASP is the
    "Pulse" variant of SAFARI, using the same wavenumber-integration kernel
    as OAST but evaluated across a frequency sweep.

    Parameters
    ----------
    executable : Path, optional
        Path to OASP binary. Auto-detected if ``None``.
    n_time_samples : int, optional
        Number of FFT time samples. Default ``4096``.
    freq_max : float, optional
        Maximum FFT frequency (Hz). ``None`` (default) derives
        ``2.5 × `` the centre frequency at ``run()`` time.
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Broadband-oriented; supports ``RunMode.BROADBAND`` (returns a
    :class:`Field`) and ``RunMode.TIME_SERIES`` (returns a
    :class:`Field` after ``synthesize_time_series``). For a
    single-cell trace use ``tf.to_time_trace(depth, range)``. Can be
    expensive (reduce ``n_time_samples`` for speed). For range-dependent
    problems, RAM is recommended.

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    Per-model: ``'ssp': 'mean'``, ``'bottom_range': 'median'`` (the
    layer stack is kept).

    Examples
    --------
    >>> from uacpy.models import OASP
    >>> oasp = OASP(n_time_samples=256, freq_max=120)
    >>> result = oasp.run(env, source, receiver)
    """

    # Declarative metadata (see PropagationModel / ModelSpec). OASP:
    # range-independent broadband wavenumber integration / pulse synthesis;
    # multi-layer fluid + elastic bottom honoured. Single spectral solve per
    # frequency → mean SSP / median bottom column represent the path.
    spec = ModelSpec(
        modes=(RunMode.COHERENT_TL, RunMode.BROADBAND, RunMode.TIME_SERIES),
        supports={'layered_bottom', 'elastic_media',
                  'rough_surface', 'rough_bottom'},
        collapse={'ssp': 'mean', 'bottom_range': 'median'},
    )
    source = 'oases'

    def __init__(
        self,
        executable: Optional[Path] = None,
        n_time_samples: int = 4096,
        freq_max: Optional[float] = None,
        freq_min: float = 0.0,
        center_frequency: Optional[float] = None,
        freq_output_increment: Optional[int] = None,
        options: Optional[str] = None,
        range_start: Optional[float] = None,
        integration_offset: float = 0.0,
        nw_samples: int = -1,
        dip_angle: Optional[float] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: float = 600.0,
        collapse: Optional[Dict[str, str]] = None,
    ):
        """
        Parameters
        ----------
        executable : Path, optional
            Path to OASP executable. Auto-detected if None.
        n_time_samples : int, optional
            Power-of-two FFT length (samples per receiver trace).
            Default 4096.
        freq_max : float, optional
            Upper edge of the OASP broadband sweep (Hz). ``None``
            (default) derives ``2.5 ×`` the centre frequency at
            ``run()`` time; a pinned value below the centre frequency
            raises at ``run()``.
        freq_min : float, optional
            Lower edge of the OASP broadband sweep (Hz). Default 0.0.
        center_frequency : float, optional
            Carrier frequency for the pulse (Hz). ``None`` defaults to
            ``source.frequencies[0]`` at ``run()`` time.
        freq_output_increment : int, optional
            OASP's ``INTF`` (Block VII, ``unoasp22.f:160``). It gates how
            often the wavenumber *integrand* is plotted
            (``unoasp22.f:591``: ``IF (MOD(JJ-LXP1,INTF).EQ.0) KPLOT=1``,
            consumed at ``unoasp22.f:678``) and does **not** decimate the
            ``.trf`` frequency axis, which always carries every bin
            ``LXP1..MX``. ``None`` → 40.
        options : str, optional
            Raw OASES option string. ``None`` defers to the writer's
            default (``'N J'``); the ``'J'`` forces a real frequency axis
            (OMEGIM=0) so the broadband ``.trf`` synthesis stays valid.
        range_start : float, optional
            First receiver range (m). ``None`` defaults to
            ``receiver.ranges.min()``.
        integration_offset : float, optional
            Wavenumber-contour offset (dB/wavelength). Default 0.
        nw_samples : int, optional
            Wavenumber sample count. ``-1`` lets OASP auto-pick.
        dip_angle : float, optional
            Fault dip angle (degrees) for the dip-slip moment source that
            option ``'4'`` selects (``unoasp22.f:993-995``). INSRC reads it
            from the source record — as the second token without ``'L'``
            (``oaseun31.f:1114``), as the seventh with it
            (``oaseun31.f:1089``). ``None`` writes 0 when ``'4'`` is
            present and raises when it is not.
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            cleanup=cleanup, timeout=timeout, collapse=collapse,
        )
        self.n_time_samples = int(n_time_samples)
        self.freq_max = float(freq_max) if freq_max is not None else None
        self.freq_min = float(freq_min)
        self.center_frequency = (
            float(center_frequency) if center_frequency is not None else None
        )
        self.freq_output_increment = (
            int(freq_output_increment)
            if freq_output_increment is not None else None
        )
        self.options = options
        self.range_start = (
            float(range_start) if range_start is not None else None
        )
        self.integration_offset = float(integration_offset)
        self.nw_samples = int(nw_samples)
        # DA, the dip angle INSRC reads only under the '4' (dip-slip) option.
        self.dip_angle = float(dip_angle) if dip_angle is not None else None

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        #
        # Keep the user's ``executable`` arg verbatim (``None`` when
        # auto-detected) so ``model.copy()`` re-resolves the binary instead of
        # re-pinning the already-resolved absolute path. The resolved path
        # lives in ``self._exe``.
        self.executable = Path(executable) if executable is not None else None
        if self.executable is None:
            self._exe = _oases_find_executable(self, 'oasp')
        else:
            self._exe = self.executable

        if not self._exe.exists():
            raise ExecutableNotFoundError('OASP', str(self._exe))

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        *,
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        output_duration: Optional[float] = None,
    ) -> Result:
        """
        Run OASP broadband wavenumber-integration computation.

        Parameters
        ----------
        env : Environment
            Ocean environment (range-independent; range-dependent
            features are collapsed with a warning)
        source : Source
            Acoustic source
        receiver : Receiver
            Receiver array
        run_mode : RunMode, optional
            ``COHERENT_TL`` (default) — extract TL at source frequency.
            ``BROADBAND`` — full H(f).
            ``TIME_SERIES`` — real pressure p(t); requires
            ``source_waveform`` + ``sample_rate``.
        source_waveform : ndarray, optional
            Source pulse for ``TIME_SERIES`` mode.
        sample_rate : float, optional
            Sampling rate of ``source_waveform`` in Hz.
        output_duration : float, optional
            Desired output duration (seconds) for ``TIME_SERIES``. When
            given, the source waveform is zero-padded internally so the
            broadband frequency grid is tight enough (``Δf =
            1/output_duration``) to span this window without DFT
            wraparound. Defaults to ``len(source_waveform)/sample_rate``.

        Returns
        -------
        result : Result
            :class:`Field` for COHERENT_TL, :class:`Field`
            for BROADBAND, :class:`Field` for TIME_SERIES.
        """
        n_time_samples = self.n_time_samples
        freq_max = self.freq_max
        freq_min = self.freq_min

        run_mode = self._resolve_run_mode(run_mode)
        if run_mode not in (RunMode.BROADBAND, RunMode.TIME_SERIES):
            # BROADBAND is covered inside ``_prepare_timeseries``; warning
            # here too would double up.
            self._warn_ignored_run_kwargs(
                run_mode,
                source_waveform=source_waveform,
                sample_rate=sample_rate,
                output_duration=output_duration,
            )
        source_waveform, frequencies = self._prepare_timeseries(
            run_mode, source, frequencies, source_waveform, sample_rate,
            output_duration,
        )

        # COHERENT_TL is a single-frequency mode (validate_inputs rejects a
        # multi-frequency Source for it). An explicit multi-element
        # ``frequencies=`` would otherwise bypass that guard — OASP would run a
        # full broadband sweep and silently return only the bin nearest the
        # source frequency. Reject it the same way, pointing at BROADBAND.
        if run_mode == RunMode.COHERENT_TL and frequencies is not None:
            if np.atleast_1d(np.asarray(frequencies)).size > 1:
                raise ConfigurationError(
                    "OASP.run(run_mode=COHERENT_TL) takes a single frequency; "
                    "got a multi-element frequencies= vector. For broadband "
                    "H(f) use RunMode.BROADBAND."
                )

        self._reject_unreadable_options()

        if frequencies is not None:
            freqs_arr = np.atleast_1d(np.asarray(frequencies, dtype=float))
            if freqs_arr.size == 0:
                raise ConfigurationError(
                    "OASP.run(frequencies=…) requires at least one positive "
                    "frequency."
                )
            # OASP rebuilds its own (n_time_samples, freq_max) grid; an
            # explicit ``frequencies=`` vector overrides ``freq_max`` to the
            # user's max and ``n_time_samples`` so the resulting OASP bins
            # span at least the requested band. The (fmin, fmax, N) triple
            # is OASP's internal language — warn if the user vector is
            # non-equispaced so they know the resulting bins won't match
            # their input one-to-one.
            fmin_user, freq_max, n_user, _ = _oases_resample_frequencies(
                freqs_arr, 'OASP',
            )
            if self.freq_min and abs(self.freq_min - fmin_user) > 1e-9:
                warnings.warn(
                    f"OASP.run(frequencies=…) sets the sweep's lower edge to "
                    f"{fmin_user:.3f} Hz, overriding the constructor's "
                    f"freq_min={self.freq_min:.3f} Hz.",
                    UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
                )
            freq_min = fmin_user
            if n_user > 1:
                # df implied by the equispaced (fmin, fmax, N) grid that
                # OASES will actually run.
                df_user = (freq_max - fmin_user) / (n_user - 1)
            else:
                df_user = float(freqs_arr[0])
            if df_user > 0:
                # OASP's bin spacing is DLFREQ = 1/(DT*NX) (unoasp22.f:237)
                # and the writer sets DT at Nyquist, 1/(2*freq_max), so
                # NX = 2*freq_max/DLFREQ samples are needed to land bins as
                # fine as the user's df. OASP requires NT = 2^M
                # (oasp.tex:129), so round that up to a power of two — never
                # down past whatever n_time_samples already asked for.
                target = max(int(n_time_samples or 0),
                             int(np.ceil(2.0 * freq_max / df_user)))
                if target > 1:
                    n_time_samples = 1 << (target - 1).bit_length()
                else:
                    n_time_samples = 2
        else:
            fc_run = (
                self.center_frequency
                if self.center_frequency is not None
                else float(np.atleast_1d(source.frequencies)[0])
            )
            if freq_max is None:
                # Band headroom above the carrier, so the pulse's upper
                # sidelobes fall inside FR2 rather than being truncated.
                # OASES ties FR2 to nothing but FR1 (oasp.tex:131), so the
                # 2.5 is uacpy's default, not a model constraint.
                freq_max = 2.5 * fc_run
            elif fc_run > freq_max:
                raise ConfigurationError(
                    f"OASP: centre frequency {fc_run:.1f} Hz exceeds the "
                    f"pinned sweep edge freq_max={freq_max:.1f} Hz — the "
                    f"sweep would never reach the requested frequency. "
                    f"Raise freq_max (or leave it None to derive "
                    f"2.5×fc), or pass frequencies= explicitly."
                )

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)
        fm = self._setup_file_manager()

        writer_kwargs: dict = {
            'integration_offset': self.integration_offset,
            'nw_samples': self.nw_samples,
            'freq_min': freq_min,
        }
        if self.center_frequency is not None:
            writer_kwargs['center_frequency'] = self.center_frequency
        if self.freq_output_increment is not None:
            writer_kwargs['freq_output_increment'] = self.freq_output_increment
        if self.options is not None:
            writer_kwargs['options'] = self.options
        if self.range_start is not None:
            writer_kwargs['range_start'] = self.range_start
        if self.dip_angle is not None:
            writer_kwargs['dip_angle'] = self.dip_angle

        try:
            base_name = 'oasp_run'
            input_file = fm.get_path(f'{base_name}.dat')
            self._log(f"Writing OASP input file: {input_file}")
            write_oasp_input(
                filepath=input_file,
                env=env,
                source=source,
                receiver=receiver,
                n_time_samples=n_time_samples,
                freq_max=freq_max,
                **writer_kwargs,
            )

            # Run executable
            proc = self._execute(base_name, fm.work_dir)
            self._reject_wavenumber_overrun(proc)
            # Option '8' renames the transfer function to <base>.dtrf
            # (oasiun23.f:837-845 `bufch='d'//trfext`); the record layout is
            # unchanged, CFFX being declared plain COMPLEX (oasiun23.f:18).
            output_file = self._require_output(
                [fm.get_path(f'{base_name}.trf'),
                 fm.get_path(f'{base_name}.dtrf'),
                 fm.get_path(f'{base_name}.plt')],
                fm.work_dir, base_name, what='a transfer-function file',
                process=proc,
            )
            self._log(f"Reading OASP output: {output_file}")
            trf_data = read_oasp_trf(output_file,
                                     receiver_depths=receiver.depths)

            transfer_func = trf_data['transfer_function']  # shape: (n_frequencies, n_range, n_depth)

            if run_mode in (RunMode.BROADBAND, RunMode.TIME_SERIES):
                # Convention: (n_depth, n_range, n_frequencies) — trailing
                # axis is the variable dim. Source axes: (freq, range, depth).
                tf_reordered = np.transpose(transfer_func, (2, 1, 0))

                result = Field(
                    data=tf_reordered,
                    coords={
                        'depth': trf_data['depths'],
                        'range': trf_data['ranges'],
                        'frequency': trf_data['freq'],
                    },
                    phase_reference='travelling_wave',
                    **self._result_kwargs(
                        source,
                        backend='oasp',
                        frequencies=trf_data['freq'],
                        center_frequency=trf_data['center_frequency'],
                        n_time_samples=n_time_samples,
                        freq_max=freq_max,
                    ),
                )
                if run_mode == RunMode.TIME_SERIES:
                    result = result.synthesize_time_series(
                        source_waveform=source_waveform,
                        sample_rate=sample_rate,
                    )
            else:
                # COHERENT_TL: pick the bin nearest the source frequency
                # and return the complex narrowband pressure (transposed
                # to the (n_depth, n_range) layout). Users get TL via
                # ``field.tl`` or ``.to_tl()``.
                freq_idx = 0
                if len(trf_data['freq']) > 1:
                    freq_diff = np.abs(trf_data['freq'] - source.frequencies[0])
                    freq_idx = np.argmin(freq_diff)

                # One frequency slice of the same .trf array the BROADBAND
                # branch returns, so it carries the same phase reference.
                p_at_freq = transfer_func[freq_idx, :, :].T.astype(
                    np.complex128)  # (n_d, n_r)

                result = Field(
                    data=p_at_freq,
                    coords={
                        'depth': trf_data['depths'],
                        'range': trf_data['ranges'],
                    },
                    phase_reference='travelling_wave',
                    **self._result_kwargs(
                        source,
                        backend='oasp',
                        frequencies=float(trf_data['freq'][freq_idx]),
                        frequencies_available=trf_data['freq'],
                        source_depth=trf_data['source_depth'],
                        center_frequency=trf_data['center_frequency'],
                        n_time_samples=n_time_samples,
                        freq_max=freq_max,
                    ),
                )

            self._attach_output_paths(
                result, fm.work_dir, base_name,
                primary_files=(('trf_file', '.trf'),),
            )

            self._log("OASP simulation complete")
            return result

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _reject_unreadable_options(self) -> None:
        """Reject raw ``options`` letters whose ``.trf`` uacpy cannot read back.

        Every case below writes a well-formed ``.trf`` that the reader parses
        without complaint but that no longer means what the axis labels say —
        an extra output component the reader flattens away, a range axis that
        is really slowness, or a spectrum carrying a complex-contour offset
        the time-series synthesis does not undo. Left unchecked they surface
        as a plausible wrong answer, so each raises here instead. Leaving
        ``options`` unset (the writer's default ``'N J'``) hits none of them
        and skips the whole check.

        OASES GETOPT (``unoasp22.f:871-872`` ``READ(1,200) OPT`` /
        ``200 FORMAT(40A1)``, scanned character by character at ``:873``)
        ignores whitespace, so ``'NJO'`` enables ``'O'`` exactly like
        ``'N J O'`` — hence the tests below are on the character set, not on
        whitespace-split tokens.
        """
        if not self.options:
            return
        opt_chars = set(str(self.options)) - set(' \t\n')

        # The .trf reader collapses the MSUFT / ISROW / NOUT axes onto the
        # first slot, so extra output components are dropped, not reported.
        # Letters that add an IOUT slot, i.e. increment NOUT and put a second
        # component in every .trf record (unoasp22.f:887-918: N→1 V→2 H→3 R→5
        # K→6 S→7), plus 'U' (DECOMP), which splits the output over NCOMPO=5
        # separate files (unoasp22.f:455-456, opened one unit apart at
        # :539-540).
        multi_axis = {'V', 'H', 'R', 'K', 'S', 'U'} & opt_chars
        if multi_axis:
            raise ConfigurationError(
                f"OASP.run: options {sorted(multi_axis)} request "
                "multi-component / decomposed output, which the .trf "
                "reader currently flattens. Pass options without "
                "these letters (default 'N J' returns scalar pressure) "
                "or read the .trf directly."
            )
        if 'O' in opt_chars:
            # 'O' moves the frequency integration onto a complex
            # contour (Im(omega) = -ln(50)·Δf, unoasp22.f:372-373). The
            # .trf reader discards that offset and synthesize_time_series
            # does not re-apply exp(-Im(omega)·t), so the time series
            # would be silently wrong. Reject rather than mis-synthesise.
            raise ConfigurationError(
                "OASP.run: option 'O' (complex frequency integration) "
                "bakes an exp(-ln(50)·Δf·t) contour into the spectrum "
                "that uacpy's time-series synthesis does not undo. "
                "Drop 'O' (the default 'N J' uses a real frequency axis)."
            )
        if 't' in opt_chars:
            # Lowercase 't' sets INTTYP=-1 (unoasp22.f:1028-1029), and
            # unoasp22.f:178-189 then overwrites the deck's R0 / RSPACE /
            # NPLOTS with a slowness axis derived from CMIN/CMAX. The
            # .trf's second coordinate is then slowness in s/m, which
            # uacpy would label 'range' in metres.
            raise ConfigurationError(
                "OASP.run: option 't' (tau-p seismograms) replaces the "
                "receiver range axis with slowness "
                "(unoasp22.f:178-189), which uacpy's Field has no "
                "coordinate for and would mislabel as range. Drop 't'."
            )
        if 'J' not in opt_chars and self.nw_samples < 1:
            # Under automatic wavenumber sampling (nw_samples < 1 →
            # AUSAMP), OASES forces the complex frequency contour
            # OMEGIM = -ln(50)·Δf unless 'J' keeps ICNTIN > 0
            # (unoasp22.f:288-296, :304-306 then :372-373). Without 'J'
            # the .trf then carries the same offset 'O' would, which the
            # time-series synthesis cannot undo.
            raise ConfigurationError(
                "OASP.run: a custom options string without 'J' enables the "
                "complex frequency contour (OMEGIM≠0) under automatic "
                "wavenumber sampling, which uacpy's time-series synthesis "
                "cannot undo. Add 'J' (the default 'N J' keeps a real "
                "frequency axis) or pin nw_samples≥1."
            )

    def _reject_wavenumber_overrun(self, process) -> None:
        """Raise when OASP integrated on more wavenumbers than NP arrays hold.

        Under automatic sampling AUTSAM sizes the wavenumber axis from the
        frequency-range product with no bound (unoasp22.f:1130
        ``NW=(WNMAX-WNMIN)/DK+1``), and — unlike OAST, which stops at
        unoast31.f:459 — OASP has no ``NWVNO.GT.NP`` test. Past NP the run
        completes, exits 0 and writes a full-size ``.trf`` holding numbers
        that are not the field (measured 3.2e27 against 1.3e-4 for the same
        geometry with NW pinned under NP). The count OASP settled on is in
        its own stdout, so read it from there rather than re-deriving AUTSAM.
        """
        counts = [int(m) for m in
                  _OASES_NWVNO_ECHO.findall(getattr(process, 'stdout', '') or '')]
        if not counts or max(counts) <= _OASES_NP:
            return
        raise ConfigurationError(
            f"OASP sampled {max(counts)} wavenumbers, past OASES' array bound "
            f"NP = {_OASES_NP} (oases/src/compar.f:37-38); OASP has no bound "
            f"check on automatic sampling (unoasp22.f:1130) so the transfer "
            f"function it wrote is not the acoustic field. The count grows "
            f"with the highest frequency times the largest receiver range.",
            remediation=(
                "Shorten receiver.ranges, lower freq_max / the source "
                "frequency, or pin nw_samples <= "
                f"{_OASES_NP} to integrate on a bounded grid."),
        )

    _FOR_FILES = {
        'FOR002': 'src',
    }
    _OUTPUT_SUFFIXES = ('.trf', '.dtrf', '.plt')
