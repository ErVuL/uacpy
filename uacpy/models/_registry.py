"""The registry of concrete propagation engines.

One line per engine: the key names the engine, the entry says where its
class lives. The entry holds module paths as strings, so the registry is
read without importing any engine. Everything that has to know "every
engine" reads it: the conformance suite
(``tests/test_engine_conformance.py``), which is parametrised over it; the
``compute_*`` wrappers, which name the engines that run a mode in their
refusals; the tests that check the public name lists against it; and every
test parametrised over the engines (``tests/conftest.py``'s
``engine_params``).

Adding an engine is one package plus one line here. An abstract
intermediate (``OASES``) and test doubles are not engines and are not
listed.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

__all__ = ['EngineEntry', 'ENGINES', 'engine_classes', 'engines_running']


@dataclass(frozen=True)
class EngineEntry:
    """Where one engine's class lives, the constructor keywords that make a
    small runnable instance of it (the knobs it requires, if any, and the
    ones that keep an example run short), the ``BoundaryProperties``
    keywords of the seabed of the example guide it runs on (empty: the
    conformance suite's Pekeris half-space) — for an engine that refuses
    that seabed (SPARC's deck carries only vacuum or rigid floors; OASS and
    OASSP need a rough one) — and the installs it runs on (``requires``):
    ``'binary'`` for every engine, plus ``'oases'`` for the separately
    licensed OASES family. Each install names the pytest marker
    ``requires_<install>`` its tests carry."""

    module: str
    class_name: str
    example_kwargs: Tuple[Tuple[str, object], ...] = ()
    example_bottom: Tuple[Tuple[str, object], ...] = ()
    requires: Tuple[str, ...] = ('binary',)

    def load(self):
        """The engine class (imports its module)."""
        return getattr(importlib.import_module(self.module), self.class_name)


#: The Pekeris half-space with a rough (0.5 m) top: OASS and OASSP scatter
#: from a rough interface and refuse a smooth seabed, whose mean field would
#: write an empty ``.rhs``.
_ROUGH_HALF_SPACE = (('acoustic_type', 'half-space'), ('sound_speed', 1700.0),
                     ('density', 1.8), ('attenuation', 0.5),
                     ('roughness', 0.5))

#: The installs an OASES engine runs on: a binary, from the separately
#: licensed OASES distribution.
_OASES_INSTALL = ('binary', 'oases')


ENGINES: Dict[str, EngineEntry] = {
    'bellhop': EngineEntry('uacpy.models.bellhop', 'Bellhop'),
    'kraken': EngineEntry('uacpy.models.kraken', 'Kraken'),
    'scooter': EngineEntry('uacpy.models.scooter', 'Scooter'),
    'sparc': EngineEntry('uacpy.models.sparc', 'SPARC',
                         example_bottom=(('acoustic_type', 'rigid'),)),
    'bounce': EngineEntry('uacpy.models.bounce', 'Bounce'),
    'ram': EngineEntry('uacpy.models.ram', 'RAM'),
    'oast': EngineEntry('uacpy.models.oases', 'OAST',
                        requires=_OASES_INSTALL),
    'oasn': EngineEntry('uacpy.models.oases', 'OASN',
                        requires=_OASES_INSTALL),
    'oasr': EngineEntry('uacpy.models.oases', 'OASR',
                        requires=_OASES_INSTALL),
    'oasp': EngineEntry('uacpy.models.oases', 'OASP',
                        requires=_OASES_INSTALL),
    # A 256-sample FFT grid keeps the example run to a few seconds.
    'oassp': EngineEntry('uacpy.models.oases', 'OASSP',
                         (('correlation_length', 5.0),
                          ('n_time_samples', 256)),
                         example_bottom=_ROUGH_HALF_SPACE,
                         requires=_OASES_INSTALL),
    'oass': EngineEntry('uacpy.models.oases', 'OASS',
                        (('correlation_length', 10.0),),
                        example_bottom=_ROUGH_HALF_SPACE,
                        requires=_OASES_INSTALL),
}


def engine_classes() -> Dict[str, type]:
    """Every registered engine class, by registry key (imports them)."""
    return {key: entry.load() for key, entry in ENGINES.items()}


def engines_running(mode) -> List[str]:
    """Class names of the registered engines whose ``spec.modes`` holds
    ``mode``, sorted — the list a refusal offers as alternatives."""
    return sorted(cls.__name__ for cls in engine_classes().values()
                  if mode in cls.spec.modes)
