"""
Custom exception hierarchy for UACPY

Provides structured error handling with helpful remediation messages.
"""

import signal
from typing import Optional

__all__ = [
    'UACPYError',
    'ExecutableNotFoundError',
    'ModelExecutionError',
    'InvalidDepthError',
    'UnsupportedFeatureError',
    'ConfigurationError',
    'DataFetchError',
    'FileFormatError',
]


def _rebuild_exc(cls, args, kwargs):
    """Reconstruct an exception from its original constructor arguments.

    The subclasses below override ``__init__`` with multi-positional /
    keyword-only signatures but store only the formatted message in
    ``self.args``; the default ``BaseException.__reduce__`` would then unpickle
    via ``cls(*self.args)`` and fail. Each provides ``__reduce__`` pointing here
    so the exception round-trips through ``pickle`` — required for
    ``run_parallel`` to return real per-job errors instead of a
    ``BrokenProcessPool``.
    """
    return cls(*args, **kwargs)


class UACPYError(Exception):
    """Base exception for all UACPY errors."""

    def __init__(self, message: str, remediation: Optional[str] = None):
        self.message = message
        self.remediation = remediation
        super().__init__(self.message)

    def __str__(self):
        msg = self.message
        if self.remediation:
            msg += f"\n\nHow to fix:\n{self.remediation}"
        return msg


class ExecutableNotFoundError(UACPYError):
    """Raised when a compiled model executable cannot be located, or is
    there but cannot be run.

    ``reason`` distinguishes the second case — a directory, an unexpanded LFS
    pointer, a wrong-architecture build, a lost execute bit. The file is
    present, so "not found" would send the user looking for something that is
    already on disk; the remediation is the same broken-install one either
    way."""

    def __init__(self, model_name: str, executable: str,
                 search_paths: Optional[list] = None,
                 *, reason: Optional[str] = None):
        if reason:
            message = f"{model_name} executable cannot be run: {executable} ({reason})"
        else:
            message = f"{model_name} executable not found: {executable}"

        search_info = ""
        if search_paths:
            search_info = "\n\nSearched in:\n" + "\n".join(f"  • {p}" for p in search_paths)

        last_step = (f"3. Or replace {executable} with a working build{search_info}"
                     if reason else
                     f"3. Or add {executable} to your PATH{search_info}")
        remediation = (
            f"1. Run installation script:\n"
            f"   ./install.sh\n\n"
            f"2. Or compile {model_name} manually\n\n"
            f"{last_step}"
        )
        super().__init__(message, remediation)
        self.model_name = model_name
        self.executable = executable
        self.search_paths = search_paths
        self.reason = reason

    def __reduce__(self):
        return (_rebuild_exc, (ExecutableNotFoundError,
                               (self.model_name, self.executable,
                                self.search_paths), {'reason': self.reason}))


class ModelExecutionError(UACPYError):
    """Raised when a model subprocess exits with a non-zero code."""

    #: What uacpy itself passes as ``return_code`` when the binary never ran at
    #: all — it would not exec, the work directory was gone, the run timed out.
    #: A process really killed by SIGHUP also reports -1, but ``_run_subprocess``
    #: starts every child in its own session, so a terminal hangup cannot reach
    #: one; -1 here is always the sentinel, and is left unnamed for that reason.
    NEVER_LAUNCHED = -1

    def __init__(self, model_name: str, return_code: int,
                 stdout: Optional[str] = None,
                 stderr: Optional[str] = None, timed_out: bool = False):
        killed_by = self._killing_signal(return_code)
        if timed_out:
            message = f"{model_name} execution timed out"
        elif killed_by is not None:
            message = f"{model_name} was killed by signal {killed_by}"
        else:
            message = f"{model_name} execution failed (exit code: {return_code})"

        details = []
        if stderr:
            details.append(f"Error output:\n{stderr}")
        if stdout:
            details.append(f"Standard output:\n{stdout}")

        if details:
            message += "\n\n" + "\n\n".join(details)

        # SIGKILL cannot be caught, so the binary had no chance to say
        # anything and its streams are silent — on Linux the usual sender is
        # the out-of-memory killer, which is what makes a big grid fail here.
        oom_note = (
            "SIGKILL cannot be caught, so the binary left no diagnostics. On "
            "Linux the usual sender is the out-of-memory killer: try a coarser "
            "grid, fewer frequencies, or more RAM (`dmesg` confirms it).\n\n"
            if killed_by == 'SIGKILL' else ""
        )
        remediation = (
            f"{oom_note}"
            f"Check that:\n"
            f"1. Input parameters are valid\n"
            f"2. {model_name} executable is compatible with your system\n"
            f"3. Environment configuration is correct"
        )
        super().__init__(message, remediation)
        self.model_name = model_name
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out

    @classmethod
    def _killing_signal(cls, return_code) -> Optional[str]:
        """Name of the signal a negative ``return_code`` encodes, or ``None``.

        ``subprocess`` reports a signalled child as ``-signum``, and the bare
        number names nothing: "exit code: -9" is the shape an out-of-memory
        kill takes, and reads as an ordinary crash in the binary.
        """
        if not isinstance(return_code, int) or return_code >= 0:
            return None
        if return_code == cls.NEVER_LAUNCHED:
            return None
        try:
            return signal.Signals(-return_code).name
        except ValueError:
            return None

    def __reduce__(self):
        return (_rebuild_exc, (ModelExecutionError,
                               (self.model_name, self.return_code,
                                self.stdout, self.stderr),
                               {'timed_out': self.timed_out}))


class InvalidDepthError(UACPYError):
    """Raised when a source or receiver depth exceeds the depth a model can
    resolve. For most models that is the water depth; for spectral solvers
    (Scooter/SPARC) it includes the sediment column, so the message says
    "resolvable depth" rather than "environment depth"."""

    def __init__(self, depth: float, max_depth: float, context: str):
        message = f"{context} depth ({depth:.1f}m) exceeds resolvable depth ({max_depth:.1f}m)"
        remediation = f"Set {context.lower()} depth to ≤ {max_depth:.1f}m"
        super().__init__(message, remediation)
        self.depth = depth
        self.max_depth = max_depth
        self.context = context

    def __reduce__(self):
        return (_rebuild_exc, (InvalidDepthError,
                               (self.depth, self.max_depth, self.context), {}))


class UnsupportedFeatureError(UACPYError):
    """Raised when a model cannot satisfy the requested feature.

    ``alternatives_label`` controls the wording of the remediation line;
    use ``'models'`` (default) when the suggestions are other model
    classes, or ``'run modes'`` when they are :class:`RunMode` values.
    """

    def __init__(
        self,
        model_name: str,
        feature: str,
        alternatives: Optional[list] = None,
        *,
        alternatives_label: str = 'models',
    ):
        message = f"{model_name} does not support: {feature}"

        remediation = None
        if alternatives:
            remediation = (
                f"Try one of these {alternatives_label} instead:\n"
                + "\n".join(f"  • {alt}" for alt in alternatives)
            )

        super().__init__(message, remediation)
        self.model_name = model_name
        self.feature = feature
        self.alternatives = alternatives
        self.alternatives_label = alternatives_label

    def __reduce__(self):
        return (_rebuild_exc, (UnsupportedFeatureError,
                               (self.model_name, self.feature,
                                self.alternatives),
                               {'alternatives_label': self.alternatives_label}))


class ConfigurationError(UACPYError):
    """Raised when user-supplied inputs to a model wrapper or core class
    fail validation (bad parameter values, illegal combinations of
    flags, missing required kwargs, malformed envs, etc.). The generic
    "bad inputs" exception across the package. Catch via
    ``except ConfigurationError`` or, more broadly, ``except UACPYError``."""
    pass


class DataFetchError(UACPYError):
    """Raised when the data layer cannot supply an environment value
    (bathymetry, SSP, sediment, wind, …) for the requested location: the
    exception of ``uacpy.data``, whatever the reason.

    That covers a remote service being unreachable or returning an error
    or malformed payload; and equally a local read that cannot answer —
    a dataset file absent from the cache, a NetCDF missing the variable,
    a grid cell masked or over land, a CSV without the expected column.
    The local cases are the majority: the ``*_local.py`` readers raise it
    without touching the network at all.

    The split is deliberate. The caller asks one question — *can I get an
    environment value here?* — and acts the same way on every no: fall
    back to another source, or ask the user for a value. Typing the causes
    apart would force a four-way ``except`` for that one decision.

    Catch via ``except DataFetchError`` or, more broadly, ``except
    UACPYError``."""
    pass


class FileFormatError(UACPYError):
    """Raised when a model I/O file (``.shd``, ``.mod``, ``.grn``, …) is
    absent, malformed, truncated, or otherwise cannot be parsed — typically a
    sign the model run failed or produced corrupt/unexpected output. Distinct
    from :class:`ConfigurationError` (bad user input) since the file is not
    something the user supplied: a missing file the *user* named is a
    ``ConfigurationError``, a missing file a *model* should have written is a
    ``FileFormatError``. Catch via ``except FileFormatError`` or, more
    broadly, ``except UACPYError``."""
    pass
