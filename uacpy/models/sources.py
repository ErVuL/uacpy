"""Provenance + licence catalogue for the native propagation engines.

The model layer wraps third-party Fortran/C programs, each with its own
authorship and licence — exactly the kind of metadata the ``data/`` layer
already catalogues for ocean datasets (:mod:`uacpy.data.sources`). This is the
model-side equivalent: one :class:`ModelSource` per upstream program, keyed in
:data:`MODEL_SOURCES`. A :class:`~uacpy.models.base.ModelSpec` references an
entry by ``id`` via its ``source`` field, so ``PropagationModel`` can surface
the citation (``model.source`` / ``model.citation``) and — mirroring the
non-commercial fetch warning in ``data/`` — emit a one-time ``UserWarning`` when
a licence-restricted engine (OASES) is instantiated.

Licence facts are taken from the vendored ``third_party/<engine>/LICENSE`` /
``README`` files, not invented; see ``third_party/MODIFICATIONS.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional

__all__ = ['ModelSource', 'MODEL_SOURCES', 'model_source']


@dataclass(frozen=True)
class ModelSource:
    """Provenance + licence metadata for one native propagation engine.

    Mirrors :class:`uacpy.data.sources.DataSource` for the model layer. The
    two booleans drive policy: ``commercial_use=False`` triggers the one-time
    instantiation warning (as CRUST1.0 does at fetch time), and
    ``redistributable=False`` records why an engine is a user-side install
    (``install.sh``) rather than bundled in the wheel/sdist.
    """

    id: str
    name: str
    authors: str
    license: str
    citation: str
    url: str
    commercial_use: bool
    redistributable: bool
    note: str = ''

    @property
    def attribution(self) -> str:
        """Short credit ``"<primary author>, <name>"`` — the model-side
        counterpart of :attr:`uacpy.data.sources.DataSource.attribution`, so a
        single credit path can render either kind. The primary author is the
        first author group (before any ``';'``) with its parenthetical
        affiliation stripped.
        """
        primary = self.authors.split(';')[0]
        primary = re.sub(r'\([^)]*\)', '', primary).strip().rstrip(',').strip()
        return f"{primary}, {self.name}"


MODEL_SOURCES: Dict[str, ModelSource] = {
    'acoustics_toolbox': ModelSource(
        id='acoustics_toolbox',
        name='Acoustics Toolbox',
        authors='Michael B. Porter (HLS Research)',
        license='GPL-3.0-or-later',
        citation='M. B. Porter, "The KRAKEN Normal Mode Program", '
                 'SACLANTCEN SM-245 (1991); Acoustics Toolbox.',
        url='http://oalib.hlsresearch.com/AcousticsToolbox/',
        commercial_use=True,      # GPL permits commercial use (copyleft applies)
        redistributable=True,
        note='Bellhop (Fortran backend), Kraken, Scooter, SPARC and Bounce. '
             'GPL-3.0 copyleft: derivatives that redistribute must stay GPL.',
    ),
    'oases': ModelSource(
        id='oases',
        name='OASES',
        authors='Henrik Schmidt (Massachusetts Institute of Technology)',
        license='Academic — no formal open licence; not redistributable',
        citation='H. Schmidt, OASES Version 3.1 User Guide and Reference '
                 'Manual, Dept. of Ocean Engineering, MIT.',
        url='https://acoustics.mit.edu/faculty/henrik/oases.html',
        commercial_use=False,
        redistributable=False,
        note='OAST / OASN / OASR / OASP / OASS / OASSP. Academic licence: '
             'the user installs it separately (install.sh --oases yes); '
             'uacpy never bundles or redistributes the binary. Verify terms '
             'before commercial use.',
    ),
    'collins_ram': ModelSource(
        id='collins_ram',
        name="Collins' RAM parabolic-equation family",
        authors='Michael D. Collins (US Naval Research Laboratory); vendored '
                'backends by B. D. Dushaw (mpiramS) and D. C. Calvo / '
                'S. Guelton (RAMSurf)',
        license='Public domain (Collins/NRL, US Government work); backends: '
                'mpiramS CC-BY-4.0, RAMSurf BSD',
        citation='M. D. Collins, "A split-step Pade solution for the parabolic '
                 'equation method", J. Acoust. Soc. Am. 93, 1736-1742 (1993).',
        url='https://oalib-acoustics.org/',
        commercial_use=True,
        redistributable=True,
        note='RAM dispatches at run() time to ramgeo (public domain), '
             'rams0.5 / ramsurf1.5 (BSD) or mpiramS (CC-BY-4.0); see each '
             'third_party/<backend>/LICENSE.',
    ),
}


def model_source(source_id: Optional[str]) -> Optional[ModelSource]:
    """The :class:`ModelSource` for ``source_id``, or ``None`` if unset."""
    if source_id is None:
        return None
    return MODEL_SOURCES[source_id]
