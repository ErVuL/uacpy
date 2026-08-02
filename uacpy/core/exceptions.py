"""
Custom exception hierarchy for UACPY

Provides structured error handling with helpful remediation messages.
"""

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

    def __init__(self, message: str, remediation: str = None):
        self.message = message
        self.remediation = remediation
        super().__init__(self.message)

    def __str__(self):
        msg = self.message
        if self.remediation:
            msg += f"\n\nHow to fix:\n{self.remediation}"
        return msg


class ExecutableNotFoundError(UACPYError):
    """Raised when a compiled model executable cannot be located."""

    def __init__(self, model_name: str, executable: str, search_paths: list = None):
        message = f"{model_name} executable not found: {executable}"

        search_info = ""
        if search_paths:
            search_info = "\n\nSearched in:\n" + "\n".join(f"  • {p}" for p in search_paths)

        remediation = (
            f"1. Run installation script:\n"
            f"   ./install.sh\n\n"
            f"2. Or compile {model_name} manually\n\n"
            f"3. Or add {executable} to your PATH{search_info}"
        )
        super().__init__(message, remediation)
        self.model_name = model_name
        self.executable = executable
        self.search_paths = search_paths

    def __reduce__(self):
        return (_rebuild_exc, (ExecutableNotFoundError,
                               (self.model_name, self.executable,
                                self.search_paths), {}))


class ModelExecutionError(UACPYError):
    """Raised when a model subprocess exits with a non-zero code."""

    def __init__(self, model_name: str, return_code: int, stdout: str = None,
                 stderr: str = None, timed_out: bool = False):
        if timed_out:
            message = f"{model_name} execution timed out"
        else:
            message = f"{model_name} execution failed (exit code: {return_code})"

        details = []
        if stderr:
            details.append(f"Error output:\n{stderr}")
        if stdout:
            details.append(f"Standard output:\n{stdout}")

        if details:
            message += "\n\n" + "\n\n".join(details)

        remediation = (
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
        alternatives: list = None,
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
    """Raised when an on-demand external-data fetch (bathymetry, SSP,
    sediment, …) fails: the remote service is unreachable, returns an
    error or malformed payload, or has no valid data at the requested
    location. Catch via ``except DataFetchError`` or, more broadly,
    ``except UACPYError``."""
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
