"""
Custom exception hierarchy for UACPY

Provides structured error handling with helpful remediation messages.
"""


class UACPYError(Exception):
    """Base exception for all UACPY errors"""

    def __init__(self, message: str, remediation: str = None):
        self.message = message
        self.remediation = remediation
        super().__init__(self.message)

    def __str__(self):
        msg = self.message
        if self.remediation:
            msg += f"\n\nHow to fix:\n{self.remediation}"
        return msg


class ModelError(UACPYError):
    """Base class for model-specific errors"""
    pass


class ExecutableNotFoundError(ModelError):
    """Model executable not found"""

    def __init__(self, model_name: str, executable: str, search_paths: list = None):
        message = f"{model_name} executable not found: {executable}"

        search_info = ""
        if search_paths:
            search_info = f"\n\nSearched in:\n" + "\n".join(f"  • {p}" for p in search_paths)

        remediation = (
            f"1. Run installation script:\n"
            f"   ./install_oalib.sh\n\n"
            f"2. Or compile {model_name} manually (see CLAUDE.md)\n\n"
            f"3. Or add {executable} to your PATH{search_info}"
        )
        super().__init__(message, remediation)
        self.model_name = model_name
        self.executable = executable


class ModelExecutionError(ModelError):
    """Model execution failed"""

    def __init__(self, model_name: str, return_code: int, stdout: str = None, stderr: str = None):
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


class EnvironmentError(UACPYError):
    """Environment configuration errors"""
    pass


class InvalidDepthError(EnvironmentError):
    """Depth validation error"""

    def __init__(self, depth: float, max_depth: float, context: str):
        message = f"{context} depth ({depth:.1f}m) exceeds environment depth ({max_depth:.1f}m)"
        remediation = f"Set {context.lower()} depth to ≤ {max_depth:.1f}m"
        super().__init__(message, remediation)
        self.depth = depth
        self.max_depth = max_depth
        self.context = context


class InvalidRangeError(EnvironmentError):
    """Range validation error"""

    def __init__(self, range_value: float, message: str):
        remediation = "Check that all range values are positive"
        super().__init__(message, remediation)
        self.range_value = range_value


class UnsupportedFeatureError(UACPYError):
    """Model doesn't support requested feature"""

    def __init__(self, model_name: str, feature: str, alternatives: list = None):
        message = f"{model_name} does not support: {feature}"

        remediation = None
        if alternatives:
            remediation = f"Try these models instead:\n" + "\n".join(f"  • {alt}" for alt in alternatives)

        super().__init__(message, remediation)
        self.model_name = model_name
        self.feature = feature
        self.alternatives = alternatives


class ConfigurationError(UACPYError):
    """Configuration file errors"""
    pass


class ValidationError(UACPYError):
    """Input validation errors"""
    pass
