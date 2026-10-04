"""Array-product result types — the spatial :class:`Covariance` and the
matched-field :class:`Replicas` — and :func:`ambiguity_field`, the one
builder of the ambiguity Field a matched-field processor returns."""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional

from uacpy.core.exceptions import ConfigurationError

from uacpy.core.results._base import Result, coordinate_axis
from uacpy.core.results.quantities import coordinate_unit
from uacpy.core._repr import count


def _copy_or_none(array):
    """A copy of ``array``, or ``None`` for ``None``."""
    return None if array is None else np.array(array)


def _check_frequency_axis(result, n_slices: int, who: str) -> None:
    """Refuse a ``frequencies`` axis whose length is not the array's
    frequency-axis length: ``n_frequencies`` reads the array, and a
    frequency label per slice is what ``frequencies`` claims to be."""
    if result.frequencies is not None and len(result.frequencies) != n_slices:
        raise ConfigurationError(
            f"{who}: frequencies holds {len(result.frequencies)} value(s) but "
            f"the array's frequency axis (axis 0) holds {n_slices}; give one "
            f"frequency per slice, or leave frequencies unset.")


def _check_receiver_positions(positions, n_receivers: int, who: str):
    """``positions`` as an owned ``(n_receivers, 3)`` float array of
    ``(x, y, z)`` metres, or ``None``; any other shape is refused."""
    if positions is None:
        return None
    rp = np.array(positions, dtype=float)
    if rp.ndim != 2 or rp.shape[1] != 3 or rp.shape[0] != n_receivers:
        raise ConfigurationError(
            f"{who}.receiver_positions: must have shape "
            f"(n_receivers={n_receivers}, 3); got {rp.shape}."
        )
    return rp


def _receiver_coords(positions) -> dict:
    """The ``receiver_x`` / ``receiver_y`` / ``receiver_z`` auxiliary
    coordinates (m) along ``receiver`` of an ``(n, 3)`` position table, or
    none."""
    if positions is None:
        return {}
    return {f'receiver_{axis}': (positions[:, i], 'm', 'receiver')
            for i, axis in enumerate('xyz')}


def _positions_from(arrays):
    """The ``(n, 3)`` position table :func:`_receiver_coords` wrote, or
    ``None``."""
    if 'receiver_x' not in arrays:
        return None
    return np.stack([arrays[f'receiver_{axis}'] for axis in 'xyz'], axis=1)


def _frequency_coord(result) -> dict:
    """The ``frequency`` coordinate of a result that labels its slices, or
    none."""
    if result.frequencies is None:
        return {}
    return {'frequency': (result.frequencies, coordinate_unit('frequency'))}


def ambiguity_field(surface, candidates, *, reference_unit: str,
                    frequencies=None, model: str = ''):
    """A matched-field ambiguity surface as a
    :class:`~uacpy.core.results.Field`, in dB re its own maximum.

    ``surface`` is a linear processor output —
    :func:`uacpy.acoustic_signal.bartlett` or
    :func:`~uacpy.acoustic_signal.mvdr` over a replica bank — indexed by the
    ``candidates`` mapping in order: one named coordinate per surface axis,
    in the Field's vocabulary (``'frequency'``, ``'depth'``, ``'range'``,
    ``'x'``, ``'y'``). The Field carries ``kind='ambiguity'``, so it plots
    with ``.plot()`` on the ambiguity colormap and ``.max()`` returns the
    estimate as a slice with its coordinates.

    The linear power the dB values are relative to is kept on the Field as
    :attr:`~uacpy.core.results.Field.reference` in ``reference_unit`` (the
    covariance's unit for an unnormalised surface, ``'1'`` for a trace- or
    max-normalised one), so ``field.reference * 10**(field.data / 10)`` is
    the linear surface again.

    A candidate with zero power is ``-inf`` dB and one :func:`mvdr` could not
    evaluate stays NaN; neither is floored, since a floor is a display choice
    (``plot_matched_field``'s ``dynamic_range_dB``, or ``vmin=`` on
    ``.plot()``).

    Parameters
    ----------
    surface : ndarray
        The linear processor output.
    candidates : mapping
        One named coordinate per surface axis, in order.
    reference_unit : str
        Unit of the linear power the dB values are relative to.
    frequencies : array_like, optional
        Frequencies (Hz) stamped on the Field.
    model : str, optional
        Model name stamped on the Field.

    Raises
    ------
    ConfigurationError
        When the surface's shape is not the candidate grid's, or it has no
        finite positive peak to be relative to.
    """
    # Imported here: this module is loaded with the results package, which
    # must not load field.py's dependencies before a Field is built.
    from uacpy.core.results.field import Field
    power = np.asarray(surface, dtype=float)
    coords = {str(name): np.atleast_1d(np.asarray(values, dtype=float))
              for name, values in dict(candidates).items()}
    grid = tuple(values.size for values in coords.values())
    if power.shape != grid:
        raise ConfigurationError(
            f"ambiguity_field: surface has shape {power.shape}, but the "
            f"candidate grid {list(coords)} is {grid}; the surface is indexed "
            f"by the candidates in order.")
    finite = power[np.isfinite(power)]
    peak = float(finite.max()) if finite.size else float('nan')
    if not (peak > 0.0):
        raise ConfigurationError(
            "ambiguity_field: the surface has no finite positive peak, so a "
            "level relative to it is undefined.")
    with np.errstate(divide='ignore'):
        level = 10.0 * np.log10(power / peak)
    return Field(data=level, coords=coords, model=model,
                 frequencies=frequencies, kind='ambiguity', unit='dB',
                 reference=peak, reference_unit=str(reference_unit))


class Covariance(Result):
    """Spatial covariance matrix ``C(f, i, j)``, as OASN writes it.

    Hydrophone × hydrophone correlation per frequency, written by OASN with
    option ``N`` to a ``.xsm`` file. The eigenvectors of ``C[ifreq]`` are
    matched-field-processing replica vectors used for signal-subspace
    detection and localization.

    Attributes
    ----------
    covariance : ndarray, shape ``(n_frequencies, n_receivers, n_receivers)``
        Complex covariance matrices.
    receiver_positions : ndarray, optional, shape ``(n_receivers, 3)``
        ``(x, y, z)`` positions in metres.
    unit : str
        The unit of ``covariance`` (``'Pa²/Hz'`` from OASN); ``''`` when
        the producer states none.

    Notes
    -----
    To extract MFP signal-subspace eigenvectors call
    ``np.linalg.eigh(cov.covariance[ifreq])`` directly.
    """

    def __init__(
        self,
        *,
        covariance: np.ndarray,
        receiver_positions: Optional[np.ndarray] = None,
        unit: str = '',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.unit = str(unit)
        # Copy on ingest so a caller mutating their source array can't silently
        # corrupt this result.
        cov = np.array(covariance)
        if cov.ndim != 3 or cov.shape[1] != cov.shape[2]:
            raise ConfigurationError(
                f"Covariance.covariance: must be 3-D (n_freq, n_rcv, n_rcv); "
                f"got shape {cov.shape}."
            )
        _check_frequency_axis(self, cov.shape[0], "Covariance")
        self.covariance = cov
        self.receiver_positions = _check_receiver_positions(
            receiver_positions, cov.shape[1], "Covariance")

    @property
    def n_frequencies(self) -> int:
        """Frequency slices held, ``covariance.shape[0]``; equal to
        ``len(frequencies)`` whenever ``frequencies`` is set (checked at
        construction), and still the slice count when it is not."""
        return int(self.covariance.shape[0])

    @property
    def n_receivers(self) -> int:
        return int(self.covariance.shape[1])

    def _repr_bits(self) -> list:
        return [count(self.n_receivers, 'receiver')]

    def to_dict(self) -> dict:
        """Serialise this covariance to plain arrays: ``covariance``
        ``(n_f, n_rcv, n_rcv)``, ``receiver_positions`` (``None`` when not
        recorded) and the identity, as :meth:`Field.to_dict` writes it.
        ``np.savez(f, **d)`` stores it; read it back with
        ``np.load(f, allow_pickle=True)`` into :meth:`from_dict`."""
        return {
            'covariance': self.covariance.copy(),
            'receiver_positions': _copy_or_none(self.receiver_positions),
            'unit': self.unit,
            **self._identity_dict(),
        }

    def _payload(self):
        return {'covariance': (self.covariance,
                               ('frequency', 'receiver', 'receiver_'),
                               self.unit)}

    def _export_attrs(self):
        return {**super()._export_attrs(), 'unit': self.unit}

    def _coords(self):
        return {**_frequency_coord(self),
                **_receiver_coords(self.receiver_positions)}

    @classmethod
    def _from_export(cls, arrays, attrs):
        return cls(covariance=arrays['covariance'],
                   receiver_positions=_positions_from(arrays),
                   unit=str(attrs.get('unit', '')),
                   **cls._identity_from_attrs(attrs, reserved=('unit',)))

    @classmethod
    def from_dict(cls, d: dict) -> "Covariance":
        """Reconstruct a :class:`Covariance` from :meth:`to_dict` output, or
        from the mapping ``np.load(f, allow_pickle=True)`` returns for a file
        written with ``np.savez(f, **cov.to_dict())``.

        Parameters
        ----------
        d : mapping
            :meth:`to_dict` output, or the mapping ``np.load`` returns for it.
        """
        d = cls._unwrap_saved(d, payload=('covariance',))
        return cls(covariance=d['covariance'],
                   receiver_positions=d.get('receiver_positions'),
                   unit=str(d.get('unit', '')),
                   **cls._identity_from_dict(d))

    def _ambiguity_candidates(self, replicas: "Replicas") -> dict:
        """The candidate grid of an ambiguity surface of ``replicas`` against
        this covariance: ``'frequency'`` first, then the replicas' own
        candidate axes. Refuses replicas of another frequency count or
        another array, and a frequency axis neither result records."""
        if replicas.n_frequencies != self.n_frequencies:
            raise ConfigurationError(
                f"Covariance MFP: frequency mismatch — "
                f"covariance has {self.n_frequencies} freq, "
                f"replicas has {replicas.n_frequencies}."
            )
        if replicas.n_receivers != self.n_receivers:
            raise ConfigurationError(
                f"Covariance MFP: receiver-count mismatch — "
                f"covariance has {self.n_receivers}, "
                f"replicas has {replicas.n_receivers}."
            )
        frequencies = (self.frequencies if self.frequencies is not None
                       else replicas.frequencies)
        if frequencies is None:
            raise ConfigurationError(
                "Covariance MFP: neither the covariance nor the replicas "
                "record their frequencies, so the ambiguity surface has no "
                "frequency axis to stand on. Build them with frequencies=.")
        return {'frequency': frequencies, **replicas.candidates}

    def _ambiguity(self, surface, candidates: dict):
        """``surface`` as the ambiguity Field over ``candidates`` (from
        :meth:`_ambiguity_candidates`), relative to its peak in this
        covariance's unit (:attr:`unit`; ``''`` when it records none)."""
        return ambiguity_field(
            surface, candidates,
            reference_unit=self.unit,
            frequencies=candidates['frequency'], model=self.model)

    def bartlett(self, replicas: "Replicas"):
        """Conventional Bartlett MFP ambiguity surface of ``replicas``
        against this covariance, per frequency:
        :func:`uacpy.acoustic_signal.bartlett` with ``normalize=None``.

        ``B(f, c) = w(f, c)ᴴ · C(f) · w(f, c)``, with ``w`` the replica
        vector at each candidate point ``c`` scaled to unit length. A
        zero-norm replica — a candidate position the forward model put no
        energy at — scores a genuine zero ("nothing matches here"), ``-inf``
        dB; :meth:`mvdr` instead leaves it undefined, NaN.

        Parameters
        ----------
        replicas : Replicas
            The candidate replica set.

        Returns
        -------
        Field
            ``kind='ambiguity'``, in dB re the surface's peak over every
            frequency and candidate, on
            ``('frequency', *replicas.candidates)``.
            ``reference`` is that peak power, in this covariance's unit.
        """
        from uacpy.acoustic_signal.beamforming import bartlett
        candidates = self._ambiguity_candidates(replicas)
        return self._ambiguity(bartlett(self.covariance, replicas.replicas),
                               candidates)

    def mvdr(self, replicas: "Replicas", *, diagonal_loading: float = 1e-6):
        """Minimum-Variance Distortionless-Response (Capon) MFP ambiguity
        surface of ``replicas`` against this covariance, per frequency:
        :func:`uacpy.acoustic_signal.mvdr` with ``normalize=None``.

        ``M(f, c) = 1 / (wᴴ · (C(f) + δ·I)⁻¹ · w)`` with
        ``δ = diagonal_loading · trace(C(f))/N``. Small loading (~1e-6)
        stabilises a rank-deficient covariance for sharp Capon peaks; larger
        loading (~0.1+) flattens the surface toward Bartlett for mismatch
        robustness. The 1e-6 default suits the full-rank covariance OASN
        writes to its ``.xsm``; :func:`uacpy.sonar.mvdr`, over a *measured*
        few-snapshot CSDM, defaults to 1e-2 instead. A frequency bin carrying
        no power is NaN throughout, with a warning, and so is a candidate
        with a zero-norm replica.

        Parameters
        ----------
        replicas : Replicas
            The candidate replica set.
        diagonal_loading : float, optional
            Loading as a fraction of ``trace(C)/N``. Default 1e-6.

        Returns
        -------
        Field
            As :meth:`bartlett`.
        """
        from uacpy.acoustic_signal.beamforming import mvdr
        candidates = self._ambiguity_candidates(replicas)
        return self._ambiguity(
            mvdr(self.covariance, replicas.replicas,
                 diagonal_loading=diagonal_loading),
            candidates)


class Replicas(Result):
    """Matched-field-processing replicas: the array response to a source at
    every candidate position.

    Frequency-domain Green's-function samples at every array element for
    every candidate source position — written by OASN with option ``R`` to a
    ``.rpo`` file, or built by :func:`uacpy.sonar.replica_bank` /
    :func:`~uacpy.sonar.replica_bank_from_field`.

    Attributes
    ----------
    replicas : ndarray, shape ``(n_frequencies, *candidate_grid, n_receivers)``
        Complex array responses, the element axis last: one row of element
        weights per candidate, the layout
        :func:`uacpy.acoustic_signal.bartlett` takes.
    candidates : dict
        The candidate grid, one named coordinate (m) per grid axis in order:
        ``{'depth', 'x', 'y'}`` for OASN, ``{'depth', 'range'}`` for a
        vertical-array replica bank.
    receiver_positions : ndarray, optional, shape ``(n_receivers, 3)``
        ``(x, y, z)`` positions in metres.

    Notes
    -----
    Feed these to :meth:`Covariance.bartlett` / :meth:`Covariance.mvdr` or
    :func:`uacpy.sonar.bartlett` / :func:`~uacpy.sonar.mvdr` for an
    ambiguity surface; each contracts a covariance against the replicas
    across the element axis.
    """

    def __init__(
        self,
        *,
        replicas: np.ndarray,
        candidates: Dict[str, np.ndarray],
        receiver_positions: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Copy on ingest so a caller mutating their source array can't silently
        # corrupt this result.
        rep = np.array(replicas)
        if not isinstance(candidates, dict) or not candidates:
            raise ConfigurationError(
                "Replicas.candidates: must be a non-empty dict of candidate "
                "axis name → coordinates (m), one per grid axis in order; got "
                f"{type(candidates).__name__}.")
        grid = {str(name): np.atleast_1d(np.array(values, dtype=float))
                for name, values in candidates.items()}
        if rep.ndim != len(grid) + 2:
            raise ConfigurationError(
                f"Replicas.replicas: must be (n_freq, *candidate_grid, "
                f"n_rcv) with one grid axis per candidate {list(grid)}, "
                f"{len(grid) + 2}-D; got shape {rep.shape}.")
        _check_frequency_axis(self, rep.shape[0], "Replicas")
        expected = tuple(values.size for values in grid.values())
        if rep.shape[1:-1] != expected:
            raise ConfigurationError(
                f"Replicas.replicas: the grid axes {rep.shape[1:-1]} must "
                f"match the candidates {list(grid)} = {expected}."
            )
        self.replicas = rep
        self.candidates = grid
        self.receiver_positions = _check_receiver_positions(
            receiver_positions, rep.shape[-1], "Replicas")

    @property
    def n_frequencies(self) -> int:
        """Frequency slices held, ``replicas.shape[0]``; equal to
        ``len(frequencies)`` whenever ``frequencies`` is set (checked at
        construction), and still the slice count when it is not."""
        return int(self.replicas.shape[0])

    @property
    def n_receivers(self) -> int:
        return int(self.replicas.shape[-1])

    @property
    def n_replica_points(self) -> int:
        return int(np.prod(self.replicas.shape[1:-1]))

    def _repr_bits(self) -> list:
        return [*(coordinate_axis(name, values)
                  for name, values in self.candidates.items()),
                count(self.n_receivers, 'receiver')]

    def to_dict(self) -> dict:
        """Serialise these replicas to plain arrays: ``replicas``
        ``(n_f, *candidate_grid, n_rcv)``, ``candidates`` (a dict of
        coordinate arrays, metres), ``receiver_positions`` (``None`` when
        not recorded) and the identity, as :meth:`Field.to_dict` writes it.
        ``np.savez(f, **d)`` stores it; read it back with
        ``np.load(f, allow_pickle=True)`` into :meth:`from_dict`."""
        return {
            'replicas': self.replicas.copy(),
            'candidates': {name: values.copy()
                           for name, values in self.candidates.items()},
            'receiver_positions': _copy_or_none(self.receiver_positions),
            **self._identity_dict(),
        }

    def _payload(self):
        return {'replicas': (self.replicas,
                             ('frequency', *self.candidates, 'receiver'), '')}

    def _coords(self):
        return {**_frequency_coord(self),
                **{name: (values, coordinate_unit(name))
                   for name, values in self.candidates.items()},
                **_receiver_coords(self.receiver_positions)}

    @classmethod
    def from_xarray(cls, obj) -> "Replicas":
        """:class:`Replicas` from the ``xarray.Dataset`` :meth:`to_xarray`
        writes: the candidate axes are the dimensions between ``frequency``
        and ``receiver``, in order.

        Parameters
        ----------
        obj : xarray.Dataset
            A dataset as :meth:`to_xarray` writes it.
        """
        from uacpy.core._export import join_complex
        obj = join_complex(obj)
        names = [str(d) for d in obj['replicas'].dims[1:-1]]
        return cls(replicas=np.asarray(obj['replicas'].values),
                   candidates={name: np.asarray(obj[name].values)
                               for name in names},
                   receiver_positions=_positions_from(
                       {name: np.asarray(c.values)
                        for name, c in obj.coords.items()}),
                   **cls._identity_from_attrs(dict(obj.attrs)))

    @classmethod
    def from_dict(cls, d: dict) -> "Replicas":
        """Reconstruct :class:`Replicas` from :meth:`to_dict` output, or from
        the mapping ``np.load(f, allow_pickle=True)`` returns for a file
        written with ``np.savez(f, **replicas.to_dict())``.

        Parameters
        ----------
        d : mapping
            :meth:`to_dict` output, or the mapping ``np.load`` returns for it.
        """
        d = cls._unwrap_saved(d, payload=('replicas',))
        return cls(replicas=d['replicas'], candidates=d['candidates'],
                   receiver_positions=d.get('receiver_positions'),
                   **cls._identity_from_dict(d))
