"""
Smoke tests for uacpy.io.env_reader.

env_reader is a partial MATLAB port — six helper functions (readsxsy, readszrz,
readr, readtheta, read_bell, topbot) still raise NotImplementedError. The
*model-launch path* never calls into this module: model wrappers WRITE .env
files via ATEnvWriter, they don't read them back. This module is for users
who want to load an existing .env into a uacpy Environment.

These tests cover the bits that *should* work and lock in the recent fix that
defined SSPStruct (previously a F821 undefined-name flake8 error). Full
.env round-trip coverage is a roadmap item — see README.md hardening list.
"""

import numpy as np
import pytest

from uacpy.io import env_reader
from uacpy.io.env_reader import SSPStruct


class TestSSPStruct:
    """SSPStruct is the return-type container for read_env_core."""

    def test_sspstruct_constructible_with_no_args(self):
        """The parser does ``SSP = SSPStruct()`` and then sets attributes —
        the no-arg constructor must work."""
        ssp = SSPStruct()
        assert ssp is not None

    def test_sspstruct_default_attributes(self):
        """Default field values mirror what the parser expects to find on a
        freshly-constructed instance before it starts filling them in."""
        ssp = SSPStruct()
        assert ssp.NMedia == 0
        assert ssp.N == []
        assert ssp.sigma == []
        assert ssp.depth == []
        assert isinstance(ssp.z, np.ndarray) and ssp.z.size == 0
        assert isinstance(ssp.c, np.ndarray) and ssp.c.size == 0
        assert isinstance(ssp.cs, np.ndarray) and ssp.cs.size == 0
        assert isinstance(ssp.rho, np.ndarray) and ssp.rho.size == 0
        assert ssp.raw == []
        assert ssp.Npts == []

    def test_sspstruct_attributes_independent_per_instance(self):
        """Each new SSPStruct must get its own list/array — guards against
        the classic mutable-default-argument bug."""
        a = SSPStruct()
        b = SSPStruct()
        a.N.append(42)
        a.z = np.array([1.0, 2.0])
        assert b.N == []
        assert b.z.size == 0


class TestModuleImports:
    """The module must import cleanly with all its references resolved.

    These would have caught the original F821 (undefined SSPStruct, undefined
    crci) bugs before they reached CI's lint job.
    """

    def test_module_imports(self):
        """env_reader imports without error."""
        assert env_reader is not None

    def test_crci_reachable(self):
        """``crci`` is imported into env_reader's namespace (used by
        read_env_core when parsing layered media)."""
        assert hasattr(env_reader, "crci")
        assert callable(env_reader.crci)

    def test_stub_helpers_raise_not_implemented(self):
        """The six unported helpers must raise NotImplementedError, not
        silently return None — locks in the documented behavior."""
        stubs = ["readsxsy", "readszrz", "readr", "readtheta", "read_bell", "topbot"]
        for name in stubs:
            fn = getattr(env_reader, name)
            with pytest.raises(NotImplementedError):
                # Each stub takes different args, so introspect & dummy them
                import inspect

                sig = inspect.signature(fn)
                fn(*([None] * len(sig.parameters)))
