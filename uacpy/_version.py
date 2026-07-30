"""Single source of truth for the package version.

``pyproject.toml`` reads this attribute via setuptools' ``dynamic`` version
directive, so the declared and the importable version cannot diverge.
"""

__version__ = '0.4.2'
