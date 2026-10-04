"""The abstract base of the OASES programs and the factory that picks one
by run mode (:meth:`OASES.for_mode`)."""

import warnings
import dataclasses
from pathlib import Path

from uacpy.models.base import PropagationModel
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.models._launch import _tail
from uacpy.core.run_settings import RunMode
from uacpy.core.exceptions import (
    ConfigurationError, IOWarning, ModelExecutionError,
    UnsupportedFeatureError,
)
from uacpy.models.oases._common import _oases_subprocess_env


class OASES(PropagationModel):
    """Public base + factory for the OASES sub-models
    (OAST/OASN/OASR/OASP/OASS/OASSP).

    Each sub-model wraps one OASES binary and differs only in its per-binary
    ``FORnnn`` unit-number map (``_FOR_FILES``) and its run modes; all share the
    one ``_execute`` below. They subclass ``OASES``, so ``isinstance(model,
    OASES)`` is True for any of them. ``OASES`` itself is abstract — instantiate
    a sub-model directly, or use :meth:`OASES.for_mode` to pick one by
    ``RunMode``."""

    _FOR_FILES: dict = {}
    # The option line is composed from the other knobs (a raw ``options=``
    # beside them is refused), so a re-run takes the knobs, not the line.
    _UNPINNED_FIELDS = frozenset({'options'})

    # Files this sub-model's binary writes: suffixes appended to
    # ``base_name``, plus the bare ``fort.NN`` names some OASES builds emit
    # instead. Both are cleared before launch so a pinned ``work_dir``
    # cannot return an earlier run's output as this run's answer.
    _OUTPUT_SUFFIXES: tuple = ()
    _OUTPUT_FORT_FILES: tuple = ()

    def __new__(cls, *args, **kwargs):
        if cls is OASES:
            raise ConfigurationError(
                "OASES is the abstract base for "
                "OAST/OASN/OASR/OASP/OASS/OASSP. "
                "Instantiate a sub-model directly, or use "
                "OASES.for_mode(run_mode=...) to pick one by RunMode."
            )
        return super().__new__(cls)

    @classmethod
    def for_mode(cls, run_mode: RunMode = RunMode.COHERENT_TL, *,
                 broadband: bool = False, reverberation: bool = False,
                 **kwargs) -> PropagationModel:
        """Instantiate the OASES sub-model that handles ``run_mode``.

        ``COHERENT_TL`` → ``OAST`` (``OASP`` when ``broadband=True``);
        ``COVARIANCE`` → ``OASN`` (``OASS`` when ``reverberation=True``);
        ``REPLICA`` → ``OASN``; ``REFLECTION`` → ``OASR``;
        ``BROADBAND``/``TIME_SERIES`` → ``OASP`` (``OASSP`` when
        ``reverberation=True``, which returns the scattered field rather than
        the coherent one); ``REVERBERATION`` → ``OASS``, which needs no flag
        because nothing else emits it. ``**kwargs`` forward to the chosen
        sub-model (a kwarg it doesn't accept raises ``TypeError``).

        Parameters
        ----------
        run_mode : RunMode, optional
            The run mode. Default ``COHERENT_TL``.
        broadband : bool, optional
            ``COHERENT_TL`` from OASP rather than OAST. Default False.
        reverberation : bool, optional
            The reverberant sub-model where one exists (see above). Default
            False.
        **kwargs
            Keywords of the chosen sub-model's constructor.
        """
        from uacpy.models.oases import (
            OASN, OASP, OASR, OASS, OASSP, OAST,
        )
        dispatch = {
            RunMode.COHERENT_TL: OASP if broadband else OAST,
            RunMode.COVARIANCE: OASS if reverberation else OASN,
            RunMode.REPLICA: OASN,
            RunMode.REFLECTION: OASR,
            RunMode.BROADBAND: OASSP if reverberation else OASP,
            RunMode.TIME_SERIES: OASSP if reverberation else OASP,
            RunMode.REVERBERATION: OASS,
        }
        target = dispatch.get(run_mode)
        if target is None:
            raise UnsupportedFeatureError(
                'OASES', str(run_mode),
                alternatives=[str(m) for m in dispatch],
                alternatives_label='run modes',
            )
        return target(**kwargs)

    def _stamp_file_result(self, result, source, *, backend: str,
                           frequencies='file', components=None, **extra):
        """Stamp a result a ``uacpy.io`` reader built from this run's output
        as the run's own: model, backend, source, ``frequencies`` (the
        reader's axis unless given), ``metadata`` reduced to
        ``n_receivers``/``title`` from the file header plus ``extra``, and
        ``components`` (the results it was built from) as its
        :attr:`~uacpy.core.results.Result.components`."""
        header = result.metadata or {}
        result.metadata = {
            key: header[key] for key in ('n_receivers', 'title')
            if key in header}
        result.metadata.update(extra)
        if components:
            result._components = {**result.components, **components}
        return self._stamp_result(
            result, source, backend=backend,
            frequencies=(result.frequencies
                         if isinstance(frequencies, str) else frequencies))

    def _execute(self, base_name: str, work_dir: Path, *, extra_env=None):
        """Run the OASES binary and return its ``CompletedProcess``.

        FOR005 stays as stdin per OASES docs. The completed process is
        returned rather than dropped: OASES writes no print file, so its
        stdout/stderr are the only record of what it did.

        ``extra_env`` adds FORnnn unit assignments that cannot live in
        ``_FOR_FILES``: ``_oases_subprocess_env`` derives every suffix from
        this run's ``base_name``, but the OASS/OASSP scattering
        post-processors read files named after the *mean-field* run's stem
        (an output for the producer, an input here). ``stale_outputs``
        clears this run's stem only, so the producer's files survive the
        pre-launch wipe — the two stems are distinct by construction.
        """
        env = _oases_subprocess_env(base_name, **self._FOR_FILES)
        if extra_env:
            env.update(extra_env)
        launch = self._prt_launch([str(self._exe)], work_dir, base_name,
                                  env=env,
                                  stale_outputs=self._OUTPUT_SUFFIXES)
        return self._launch_binary(dataclasses.replace(
            launch,
            stale_outputs=(tuple(self._OUTPUT_FORT_FILES)
                           + launch.stale_outputs),
            checks=launch.checks + (self._raise_on_stdout_fatal,
                                    self._warn_on_stdout_warnings)))

    #: Terminal lines of the OASES fatals that end in a bare ``STOP`` —
    #: exit 0, nothing on stderr — so the base class's stop-form scan cannot
    #: see them: CHKSOL (``oaseun31.f:2209-2221``), SYSADDMEM's caller
    #: (``oashun21.f:452-464``), DA_ERRMSG (``oashun21.f:716-725``), PINIT1's
    #: source/receiver order checks (``oaseun31.f:1436-1455``), MORDER
    #: (``oasiun23.f:805-806``) and oassp2's array-bound stops on the mean
    #: field's wavenumber count — ``>>> NKR too large in CALSRP<<<``
    #: (``oasvun31.f:41``), ``>>> NKMEAN too large in CALSRP<<<`` (``:62``)
    #: and the eight lines like them at ``:287``, ``:320``, ``:929-949`` and
    #: ``:1319-1352``, each a ``write(6,*)`` before a bare ``stop``, and all
    #: naming their routine as ``too large in CALS…``. Each is printed only
    #: on its fatal path.
    _STDOUT_FATAL_MARKERS = (
        'EXECUTION TERMINATED',
        '>>> FATAL',
        'out of order ***',
        'UNKNOWN SOURCE TYPE',
        'no availible logical unit',
        'too large in CALS',
    )

    def _raise_on_stdout_fatal(self, result) -> None:
        """Raise when the binary printed one of OASES' bare-STOP fatals.

        Those stops come after the output file is opened, and OASP's and
        OASR's headers are already written by then, so the file is present
        and non-empty; left alone, the reader would report it as a corrupt
        file and the diagnosis OASES printed would be lost.
        """
        text = getattr(result, 'stdout', None) or ''
        hits = [line.strip() for line in text.splitlines()
                if any(m in line for m in self._STDOUT_FATAL_MARKERS)]
        if not hits:
            return
        raise ModelExecutionError(
            self.model_name, return_code=0, stdout=_tail(text),
            stderr=(f"{self.model_name} stopped with a fatal error (exit "
                    f"status 0): " + ' | '.join(dict.fromkeys(hits))),
        )

    def _warn_on_stdout_warnings(self, result) -> None:
        """Surface the binary's own non-fatal warnings from stdout.

        The base class's pass reads ``<base_name>.prt``, which the Acoustics
        Toolbox writes and OASES does not — so without this, six model classes
        drop every diagnostic their binary emits. The one this was measured on
        is the only check of its kind anywhere in the pipeline: an elastic
        half-space with ``(cs/cp)**2 > 0.75`` makes ``oaseun31.f:204-207``
        print ``>>>>>WARNING: UNPHYSICAL SPEED RATIO``, and nothing on the
        Python side tests that ratio.

        Matched narrowly: OASES prefixes ordinary progress traces with the
        same ``>>>`` (``>>> Entering SOLDIS <<<``, ``>>> Done, CPU=``), so
        only a ``>``-prefixed line that also says *warning* is forwarded.
        These are the solver's words, passed through verbatim rather than
        reinterpreted — the same contract as the ``.prt`` pass. Fatals are
        already raised by ``_raise_on_fortran_fatal`` and quoted by
        ``_require_output``.
        """
        text = getattr(result, 'stdout', None)
        if not text:
            return
        seen, lines = set(), []
        for raw in text.splitlines():
            line = raw.strip()
            if (line.startswith('>') and 'warning' in line.lower()
                    and line not in seen):
                seen.add(line)
                lines.append(line)
        if lines:
            joined = "\n  ".join(lines)
            warnings.warn(
                f"{self.model_name} reported {len(lines)} non-fatal "
                f"warning(s) on stdout:\n  {joined}",
                IOWarning, skip_file_prefixes=USER_FRAME_SKIP)
