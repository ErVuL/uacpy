"""The repr of every public class: one line, ``ClassName(...)``, built by
the helpers in :mod:`uacpy.core._repr` (docs/DEV.md §5.2).

:data:`EXPECTED` pins the exact string of a realistic instance of each
class; :func:`test_every_public_class_prints_one_line_without_a_nested_constructor`
holds every class the package exports to the rule, and fails on a public
class that has no instance here."""

import importlib
import inspect
import pkgutil
import re
import warnings

import numpy as np
import pytest

import uacpy
from uacpy.core.run_settings import EngineSettings, RunSettings


def _instances():
    """Every case, by id: a builder of one realistic instance."""
    import uacpy
    from uacpy.core.boundary import BoundaryProperties, SedimentLayer
    from uacpy.core.bottom import Bottom, SeabedColumn
    from uacpy.core.surface import Surface
    from uacpy.core.ssp import SoundSpeedProfile
    from uacpy.core.bathymetry import Bathymetry
    from uacpy.core.altimetry import Altimetry
    from uacpy.core import absorption as A
    from uacpy.core.provenance import DataProvenance, DataSource
    from uacpy.core.results import (Field, Arrivals, Modes, Rays,
                                    GreensFunction, Covariance, Replicas,
                                    ReflectionCoefficient, ResultStack)
    from uacpy.core.results.modes import MediaTable
    from uacpy.core.results.speeds import SoundSpeeds
    from uacpy.core import run_settings as RS
    from uacpy.acoustic_signal import (bands, beamforming, cqt, detect, frf,
                                       gathers, spectral, timefreq,
                                       delay_profile)
    from uacpy.comms import channel, janus, link, transceiver
    from uacpy.noise import ambient
    from uacpy.sonar import bottom_scattering, sonar_equation

    z12 = np.linspace(5.0, 95.0, 12)
    r20 = np.linspace(100.0, 10000.0, 20)
    f12 = np.linspace(100.0, 1000.0, 12)
    t64 = np.arange(64) / 1000.0
    sand = BoundaryProperties.from_preset('sand')
    src = DataSource(id='gebco', name='GEBCO 2024 Grid', used_for='bathymetry',
                     license='public domain', attribution='GEBCO Compilation Group',
                     citation='GEBCO 2024', url='https://www.gebco.net',
                     commercial_use=True)
    prov = DataProvenance(source=src, data_point=(43.0, 5.0),
                          requested_point=(43.01, 5.02), product='GEBCO_2024')

    c = {}
    # carriers
    c['Source'] = lambda: uacpy.Source(depths=20.0, frequencies=200.0)
    c['Source_multi'] = lambda: uacpy.Source(
        depths=[10.0, 20.0], frequencies=f12, weights=[1.0, -1.0])
    c['Receiver'] = lambda: uacpy.Receiver(depths=[10.0, 50.0], ranges=r20)
    c['Receiver_grid'] = lambda: uacpy.Receiver(depths=z12, ranges=r20)
    c['SoundSpeedProfile'] = lambda: SoundSpeedProfile(
        depths=np.array([0.0, 50.0, 100.0]),
        sound_speed=np.array([1520.0, 1495.0, 1500.0]), kind='measured')
    c['Bathymetry'] = lambda: Bathymetry(ranges=np.array([0.0, 5000.0, 10000.0]),
                                         depths=np.array([100.0, 150.0, 200.0]))
    c['Altimetry'] = lambda: Altimetry(ranges=np.linspace(0, 1000, 11),
                                       heights=np.linspace(-0.5, 0.5, 11))
    c['BoundaryProperties'] = lambda: sand
    c['BoundaryProperties_vacuum'] = lambda: BoundaryProperties('vacuum')
    c['SedimentLayer'] = lambda: SedimentLayer(thickness=10.0, sound_speed=1600.0,
                                               density=1.7, attenuation=0.5,
                                               name='mud')
    c['SeabedColumn'] = lambda: SeabedColumn(
        layers=[SedimentLayer(thickness=10.0, sound_speed=1600.0, density=1.7)],
        halfspace=sand)
    c['Bottom'] = lambda: Bottom(columns=[SeabedColumn(layers=[], halfspace=sand)])
    c['Bottom_rd'] = lambda: Bottom(
        columns=[SeabedColumn(layers=[], halfspace=sand)] * 3,
        ranges=np.array([0.0, 2000.0, 5000.0]))
    c['Surface'] = lambda: Surface(nodes=[BoundaryProperties('vacuum')])
    c['Thorp'] = lambda: A.Thorp()
    c['FrancoisGarrison'] = lambda: A.FrancoisGarrison(10.0, 35.0, 8.0)
    c['ConstantAbsorption'] = lambda: A.ConstantAbsorption(0.01)
    c['BiologicalLayer'] = lambda: A.BiologicalLayer(z_top_m=100.0, z_bottom_m=300.0,
                                                     f0_hz=5000.0, Q=10.0, a0=0.5)
    c['Biological'] = lambda: A.Biological(layers=[A.BiologicalLayer(
        z_top_m=100.0, z_bottom_m=300.0, f0_hz=5000.0, Q=10.0, a0=0.5)])
    c['AbsorptionCoefficient'] = lambda: A.FrancoisGarrison(
        10.0, 35.0, 8.0).table(f12, units='dB/km')
    c['Environment'] = lambda: uacpy.Environment(
        bathymetry=100.0, ssp=1500.0, bottom='sand', name='shelf')
    c['Environment_full'] = lambda: uacpy.Environment(
        bathymetry=[(0.0, 100.0), (10000.0, 200.0)], ssp=1500.0,
        bottom='sand', absorption=A.FrancoisGarrison(10.0, 35.0, 8.0),
        name='slope', water_density=1.026)
    c['DataSource'] = lambda: src
    c['DataProvenance'] = lambda: prov
    # results
    c['Field'] = lambda: Field(
        data=np.ones((12, 20), complex), coords={'depth': z12, 'range': r20},
        kind='pressure', unit='Pa', model='Bellhop', frequencies=200.0)
    c['Field_pinned'] = lambda: Field(
        data=np.ones(20), coords={'range': r20}, pinned={'depth': 50.0},
        kind='level', unit='dB', model='Kraken', frequencies=200.0)
    c['Arrivals'] = lambda: Arrivals(
        arrivals=[{'delay': 0.1}, {'delay': 0.2}],
        receiver_depths=np.array([50.0]), receiver_ranges=np.array([1000.0]),
        model='Bellhop', frequencies=100.0)
    c['Modes'] = lambda: Modes(k=np.array([0.1 + 0j, 0.2 + 0j]), phi=np.zeros((12, 2)),
                               depths=z12, model='Kraken', frequencies=25.0)
    c['Rays'] = lambda: Rays(rays=[{'r': [0.0], 'z': [0.0]}] * 41,
                             model='Bellhop', frequencies=100.0)
    c['Rays_eigen'] = lambda: Rays(rays=[{'r': [0.0], 'z': [0.0]}], is_eigen=True,
                                   receiver_depths=np.array([60.0]),
                                   receiver_ranges=np.array([3000.0]),
                                   model='Bellhop', frequencies=100.0)
    c['ReflectionCoefficient_broadband'] = lambda: ReflectionCoefficient(
        angles=np.array([0.0, 45.0, 90.0]), magnitude=np.full((3, 2), 0.5),
        phase=np.zeros((3, 2)), model='OASR', frequencies=np.array([50.0, 100.0]))
    c['GreensFunction'] = lambda: GreensFunction(
        data=np.zeros((1, 1, 2, 256), complex), phase_speeds=np.linspace(1400, 1e4, 256),
        receiver_depths=np.array([50.0, 60.0]), source_depths=np.array([20.0]),
        model='Scooter', frequencies=100.0)
    c['Covariance'] = lambda: Covariance(covariance=np.eye(8, dtype=complex)[None],
                                         model='OASN', frequencies=100.0)
    c['Replicas'] = lambda: Replicas(
        replicas=np.zeros((1, 10, 8), complex),
        candidates={'depth': np.linspace(10, 100, 10)}, model='OASN',
        frequencies=100.0)
    c['ReflectionCoefficient'] = lambda: ReflectionCoefficient(
        angles=np.linspace(0, 90, 91), magnitude=np.ones(91), phase=np.zeros(91),
        model='Bounce', frequencies=50.0)
    def _stack():
        def slab(z):
            return Field(data=np.ones((1, 1), dtype=complex),
                         coords={'depth': np.array([10.0]), 'range': np.array([100.0])},
                         model='Bellhop', frequencies=100.0,
                         source_depths=np.array([z]))
        return ResultStack(slabs=[slab(10.0), slab(20.0)], coordinate=[10.0, 20.0])
    c['ResultStack'] = _stack
    c['MediaTable'] = lambda: MediaTable(water_density=1.0, tops=np.array([100.0]),
                                         densities=np.array([1.9]), bottom_depth=100.0,
                                         halfspace_density=1.9)
    c['SoundSpeeds'] = lambda: SoundSpeeds(surface=1520.0, water_min=1495.0,
                                           water_max=1520.0, waveguide_min=1495.0,
                                           waveguide_max=1650.0)
    # records
    c['TimeSettings'] = lambda: RS.TimeSettings(source_waveform=np.zeros(128),
                                                sample_rate=8000.0, output_duration=1.0)
    c['WaveguideSpeeds'] = lambda: RS.WaveguideSpeeds(1495.0, 1650.0)
    c['OutputSpec'] = lambda: RS.OutputSpec('Field', kind='pressure', unit='Pa',
                                            coherent=True)
    c['Notice'] = lambda: RS.Notice('rays clipped', 'rays clipped at 80 deg')
    # signal
    c['BandLevels'] = lambda: bands.BandLevels(centres=np.array([125.0, 250.0, 500.0, 1000.0, 2000.0]),
                                                levels=np.array([80.0, 78.0, 75.0, 72.0, 70.0]),
        lower=np.array([111.0, 223, 445, 891, 1782]), upper=np.array([140.0, 281, 561, 1122, 2245]),
        band_type='octave', ref=1e-6)
    c['SpectralEstimate'] = lambda: spectral.SpectralEstimate(
        frequencies=np.linspace(0, 4000, 257), power=np.ones(257))
    c['ProbabilisticSpectralEstimate'] = lambda: spectral.ProbabilisticSpectralEstimate(
        frequencies=np.linspace(0, 4000, 257), level_edges=np.linspace(40, 120, 81),
        pdf=np.zeros((257, 80)))
    c['SpectrogramResult'] = lambda: timefreq.SpectrogramResult(
        frequencies=np.linspace(0, 4000, 129), times=np.linspace(0, 1, 63),
        power=np.zeros((129, 63)))
    c['CWTResult'] = lambda: timefreq.CWTResult(
        frequencies=np.geomspace(100, 4000, 32), times=t64,
        coefficients=np.zeros((32, 64), complex))
    c['WignerVilleResult'] = lambda: timefreq.WignerVilleResult(
        frequencies=np.linspace(0, 4000, 64), times=t64, distribution=np.zeros((64, 64)))
    c['Cepstrum'] = lambda: timefreq.Cepstrum(quefrencies=t64, cepstrum=np.zeros(64))
    c['ComplexCepstrum'] = lambda: timefreq.ComplexCepstrum(cepstrum=np.zeros(64), delay=3)
    c['CQTResult'] = lambda: cqt.CQTResult(frequencies=np.geomspace(100, 4000, 48),
                                           coefficients=np.zeros(48, complex))
    c['CQSpectrogramResult'] = lambda: cqt.CQSpectrogramResult(
        frequencies=np.geomspace(100, 4000, 48), times=t64, power=np.zeros((48, 64)))
    c['AmbiguityResult'] = lambda: detect.AmbiguityResult(
        delays_s=np.linspace(-0.01, 0.01, 201), doppler_hz=np.linspace(-50, 50, 101),
        amplitude=np.zeros((101, 201)))
    c['FRFResult'] = lambda: frf.FRFResult(frequencies=f12,
                                           transfer_function=np.ones(12, complex), method='h1')
    c['FKResult'] = lambda: gathers.FKResult(
        frequencies=np.linspace(0, 500, 65), wavenumbers=np.linspace(-0.5, 0.5, 32),
        power=np.zeros((65, 32)), spectrum=np.zeros((65, 32), complex), scaling='density')
    c['RadonResult'] = lambda: gathers.RadonResult(
        moveout=np.linspace(0, 1e-3, 21), taus=t64, panel=np.zeros((21, 64)),
        kind='parabolic')
    c['TauPResult'] = lambda: gathers.TauPResult(
        slownesses=np.linspace(0, 1e-3, 21), taus=t64, panel=np.zeros((21, 64)))
    c['BeamformResult'] = lambda: beamforming.BeamformResult(
        snr=np.zeros(181), angles=np.linspace(-90, 90, 181), peak_snr=12.5)
    c['BeamformedField'] = lambda: beamforming.BeamformedField(
        response=np.zeros((181, 12)), angles=np.linspace(-90, 90, 181),
        element_power=np.ones(12), frequencies=f12)
    c['Snapshots'] = lambda: beamforming.Snapshots(frequency=500.0,
                                                   data=np.zeros((8, 16), complex))
    c['ChannelRegime'] = lambda: delay_profile.ChannelRegime(
        coherence_bandwidth_hz=250.0, signal_bandwidth_hz=1000.0,
        rms_delay_spread_s=0.004, symbol_duration_s=0.001,
        frequency_selective=True, isi_symbols=4.0, convention='0.5')
    # comms
    c['ChannelTaps'] = lambda: channel.ChannelTaps(
        taps=np.ones(12, complex), delays_s=np.linspace(0.66, 0.6655, 12),
        symbol_rate=2000.0, fc=12000.0, sps=4, first_arrival_s=0.66)
    c['BerCurve'] = lambda: link.BerCurve(ebn0_dB=np.arange(0.0, 12.0, 2.0),
                                          ber=np.logspace(-1, -6, 6), scheme='qpsk', n_bits=100000)
    c['JanusReception'] = lambda: janus.JanusReception(bits=np.zeros(64, int), crc_ok=True)
    # noise / sonar
    c['NoiseComponents'] = lambda: ambient.NoiseComponents(
        total=np.ones(12), wind=np.ones(12), shipping=np.ones(12), rain=None,
        thermal=np.ones(12), turbulence=np.ones(12))
    c['WenzNoise'] = lambda: ambient.WenzNoise(frequencies=f12, wind_speed_kn=10.0)
    c['BottomParameters'] = lambda: bottom_scattering.BottomParameters(
        density_ratio=1.845, speed_ratio=1.178, loss_parameter=0.0122,
        volume_parameter=0.002, spectral_strength=0.0041, spectral_exponent=3.25)
    # models
    c['Bellhop'] = lambda: uacpy.Bellhop()
    c['Bellhop_knobs'] = lambda: uacpy.Bellhop(beam_type='B', n_beams=500)
    c['RAM'] = lambda: uacpy.RAM()

    # data / io / misc records
    from uacpy.data._geo import AlongTrack
    from uacpy.data.argo import ArgoProfile
    from uacpy.data.bathymetry import BathyGrid
    from uacpy.data.crust1_local import Crust1Profile
    from uacpy.data.glodap_local import PHProfile
    from uacpy.data.sediment import SeabedSample
    from uacpy.data.sound_speed import TSProfile
    from uacpy.data.waves import SeaStateRecord
    from uacpy.io.bathy_io import BoundaryTable
    from uacpy.io.oalib_reader import ShdFile, SspTable
    from uacpy.io.ramsurf_reader import PeGrid
    from uacpy.models.provenance import MODEL_PROVENANCE
    from uacpy import parallel
    from uacpy.comms import coding, constellations, equalize
    z5 = np.array([0.0, 10.0, 50.0, 100.0, 500.0])
    c['AlongTrack'] = lambda: AlongTrack(ranges=r20, lats=np.linspace(43, 43.1, 20),
        lons=np.full(20, 5.0), data=np.linspace(100, 200, 20), unit='m',
        quantity='depth', provenance=prov)
    c['ArgoProfile'] = lambda: ArgoProfile(platform='6902746', cycle=12, direction='A',
        lat=43.0, lon=5.0, distance_km=12.3, time='2024-05-01', data_mode='D',
        pressure_dbar=z5, temperature=np.linspace(20, 13, 5),
        salinity=np.full(5, 38.0), provenance=prov)
    c['BathyGrid'] = lambda: BathyGrid(lats=np.linspace(43, 43.1, 11),
        lons=np.linspace(5, 5.2, 21), depths=np.full((11, 21), 100.0), provenance=prov)
    c['Crust1Profile'] = lambda: Crust1Profile(water_depth=2500.0, sediment_thickness=300.0,
        layer_names=('upper sediments', 'middle sediments'), thickness=np.array([100.0, 200.0]),
        sound_speed=np.array([1600.0, 1800.0]), shear_speed=np.array([0.0, 400.0]),
        density=np.array([1.7, 1.9]), provenance=prov)
    c['PHProfile'] = lambda: PHProfile(depths=z5, ph=np.linspace(8.1, 7.9, 5),
        ph_scale='total', provenance=prov)
    c['SeabedSample'] = lambda: SeabedSample(grain_size_phi=2.5, material='sand',
        folk_class='S', folk_class_scheme='folk', sample_point=(43.0, 5.0),
        distance_km=3.2, provenance=prov)
    c['TSProfile'] = lambda: TSProfile(depths=z5, temperature=np.linspace(20, 13, 5),
        salinity=np.full(5, 38.0), temperature_kind='in-situ', provenance=prov)
    c['SeaStateRecord'] = lambda: SeaStateRecord(hs=1.5, tp=7.0, provenance=prov)
    c['BoundaryTable'] = lambda: BoundaryTable(kind='bathymetry', path='env.bty',
        interpolation='L', ranges=np.array([0.0, 5000.0]), depths=np.array([100.0, 200.0]))
    c['ShdFile'] = lambda: ShdFile(title='shelf', plot_type='rectilin  ',
        frequencies=np.array([200.0]), source_frequency=200.0,
        stabilizing_attenuation=0.0, bearings=np.array([0.0]), source_x=None,
        source_y=None, source_depths=np.array([20.0]), receiver_depths=z12,
        receiver_ranges=r20, pressure=np.zeros((1, 1, 1, 12, 20), complex),
        pressure_frequency=200.0)
    c['SspTable'] = lambda: SspTable(path='env.ssp', ranges=np.array([0.0, 5000.0]),
        sound_speed=np.full((3, 2), 1500.0))
    c['PeGrid'] = lambda: PeGrid(quantity='transmission_loss', unit='dB', ranges=r20, depths=z12,
        data=np.zeros((12, 20)))
    c['ModelProvenance'] = lambda: MODEL_PROVENANCE['acoustics_toolbox']
    c['ParallelResult'] = lambda: parallel.ParallelResult(results=[None, None],
        errors={}, labels=[10.0, 20.0], coordinate_name='source_depth')
    c['SonarBudget'] = lambda: sonar_equation.SonarBudget(mode='passive',
        source_level_dB=180.0, detection_threshold_dB=10.0, noise_level_dB=60.0,
        directivity_index_dB=15.0)
    c['JanusPacket'] = lambda: janus.JanusPacket()
    c['LinkResult'] = lambda: link.LinkResult(ber=1e-3, evm=0.12, scheme='qpsk',
        ebn0_dB=10.0, tx_symbols=np.ones(1000, complex), rx_symbols=np.ones(1000, complex))
    c['ReceiverDiagnostics'] = lambda: transceiver.ReceiverDiagnostics(
        bits=np.zeros(2000, int), symbols=np.ones(1000, complex), mse=None,
        sync_metric=np.zeros(4000), start=120)
    c['FRF'] = lambda: frf.FRF()
    c['ConvCode'] = lambda: coding.ConvCode()
    c['Modulator'] = lambda: constellations.Modulator('qpsk')
    c['DFE'] = lambda: equalize.DFE()
    c['Transmitter'] = lambda: transceiver.Transmitter('qpsk')
    c['CommsReceiver'] = lambda: transceiver.CommsReceiver('qpsk')
    c['OFDMTransmitter'] = lambda: transceiver.OFDMTransmitter('qpsk')
    c['OFDMReceiver'] = lambda: transceiver.OFDMReceiver('qpsk')
    def _settings(model, bottom='sand'):
        env = uacpy.Environment(bathymetry=100.0, ssp=1500.0, bottom=bottom)
        return model.run_settings(env, uacpy.Source(depths=20.0, frequencies=200.0),
                                  uacpy.Receiver(depths=z12, ranges=r20))
    c['RunSettings'] = lambda: _settings(uacpy.Bellhop())
    c['BellhopSettings'] = lambda: _settings(uacpy.Bellhop()).engine
    c['RamSettings'] = lambda: _settings(uacpy.RAM()).engine
    c['RamGrid'] = lambda: _settings(uacpy.RAM()).engine.grids[0]
    c['ScooterSettings'] = lambda: _settings(uacpy.Scooter()).engine
    c['KrakenSettings'] = lambda: _settings(uacpy.Kraken()).engine
    c['KrakenLaunch'] = lambda: _settings(uacpy.Kraken()).engine.launches[0]
    c['ModelSpec'] = lambda: uacpy.Bellhop.spec

    from uacpy.models import oases as _oases
    from uacpy.io.oalib_reader import FlpFile, Flp3dFile, RtsFile, Ssp3dFile
    from uacpy.io.mpirams_reader import PsifFile
    from uacpy.io.oases_reader import OasesRhsHeader
    from uacpy.io.oases_writer import OasnNoise, OasnReplicaGrid
    c['Surface_rd'] = lambda: Surface(nodes=[BoundaryProperties('vacuum'), sand],
                                      ranges=np.array([0.0, 5000.0]))
    c['Bounce'] = lambda: uacpy.Bounce()
    c['Kraken'] = lambda: uacpy.Kraken()
    c['Scooter'] = lambda: uacpy.Scooter()
    c['SPARC'] = lambda: uacpy.SPARC()
    c['OASN'] = lambda: _oases.OASN()
    c['OASP'] = lambda: _oases.OASP()
    c['OASR'] = lambda: _oases.OASR()
    c['OASS'] = lambda: _oases.OASS(correlation_length=10.0, rms_roughness=0.5)
    c['OASSP'] = lambda: _oases.OASSP(correlation_length=10.0, rms_roughness=0.5)
    c['OAST'] = lambda: _oases.OAST()
    c['BounceSettings'] = lambda: _settings(uacpy.Bounce()).engine
    c['SparcSettings'] = lambda: uacpy.SPARC().run_settings(
        uacpy.Environment(bathymetry=100.0, ssp=1500.0, bottom=BoundaryProperties('rigid')),
        uacpy.Source(depths=20.0, frequencies=200.0),
        uacpy.Receiver(depths=z12, ranges=r20)).engine
    c['OASNSettings'] = lambda: _settings(_oases.OASN()).engine
    c['OASPSettings'] = lambda: _settings(_oases.OASP()).engine
    c['OASRSettings'] = lambda: _settings(_oases.OASR()).engine
    rough = BoundaryProperties('half-space', sound_speed=1650.0, density=1.9,
                               attenuation=0.8, roughness=0.5)
    c['OASSSettings'] = lambda: _settings(_oases.OASS(correlation_length=10.0, rms_roughness=0.5), rough).engine
    c['OASSPSettings'] = lambda: _settings(_oases.OASSP(correlation_length=10.0, rms_roughness=0.5, n_wavenumbers=2048), rough).engine
    c['OASTSettings'] = lambda: _settings(_oases.OAST()).engine
    c['Job'] = lambda: parallel.Job(model=uacpy.Bellhop(), env=None, source=None,
                                    receiver=None, label=20.0)
    c['FlpFile'] = lambda: FlpFile(title='shelf', option='RA', component='P', n_modes=12,
        profile_ranges=np.array([0.0]), source_depths=np.array([20.0]),
        receiver_depths=z12, receiver_ranges=r20, receiver_range_offsets=np.zeros(12))
    c['Flp3dFile'] = lambda: Flp3dFile(title='shelf', option='STD', method='GBT',
        tesselation_check=False, sbp_flag='O', n_modes=12, source_x=0.0, source_y=0.0,
        source_depths=np.array([20.0]), receiver_depths=z12, receiver_ranges=r20,
        bearings=np.linspace(0, 350, 36), node_x=np.zeros(3), node_y=np.zeros(3),
        node_mode_files=('a', 'b', 'c'), elements=np.zeros((1, 3), int))
    c['RtsFile'] = lambda: RtsFile(title='shelf', positions=np.zeros((2, 2)),
        times=np.linspace(0, 1, 1001), pressure=np.zeros((1001, 2)))
    c['Ssp3dFile'] = lambda: Ssp3dFile(n_x=3, n_y=3, n_z=12, x=np.linspace(0, 1000, 3),
        y=np.linspace(0, 1000, 3), z=z12, sound_speed=np.full((12, 3, 3), 1500.0))
    c['PsifFile'] = lambda: PsifFile(n_samples=1024, c0=1500.0, water_min=1480.0,
        sample_rate=1024.0, q_factor=2.0, frequencies=f12, depths=z12, ranges=r20,
        pe_field=np.zeros((12, 12, 20), complex))
    c['OasesRhsHeader'] = lambda: OasesRhsHeader(n_time_samples=1024, freq_min=50.0,
        freq_max=350.0, time_step=0.001, frequency=200.0, source_layer=2,
        n_wavenumbers=4096, interface=0)
    c['OasnNoise'] = lambda: OasnNoise(surface_level=60.0, white_level=40.0)
    c['OasnReplicaGrid'] = lambda: OasnReplicaGrid(z=(10.0, 100.0, 10), c_low=1400.0,
                                                   c_high=1e8)
    return c



EXPECTED = {
    'Source':
        'Source(depth 20 m, frequency 200 Hz, point)',
    'Source_multi':
        'Source(depths [10, 20] m, 12 frequencies 100–1000 Hz, point, weights [1, -1])',
    'Receiver':
        'Receiver(depths [10, 50] m, 20 ranges 100–10000 m)',
    'Receiver_grid':
        'Receiver(12 depths 5–95 m, 20 ranges 100–10000 m)',
    'SoundSpeedProfile':
        'SoundSpeedProfile(measured, depths [0, 50, 100] m, c=1495–1520 m/s)',
    'Bathymetry':
        'Bathymetry(ranges [0, 5000, 10000] m, depth=100–200 m)',
    'Altimetry':
        'Altimetry(11 ranges 0–1000 m, sea-surface height=-0.5–0.5 m)',
    'BoundaryProperties':
        "BoundaryProperties('sand' half-space, cp=1650 m/s, ρ=1.9513 g/cm³, α=0.8 dB/λ)",
    'BoundaryProperties_vacuum':
        'BoundaryProperties(vacuum)',
    'SedimentLayer':
        "SedimentLayer('mud', thickness=10 m, cp=1600 m/s, ρ=1.7 g/cm³, α=0.5 dB/λ)",
    'SeabedColumn':
        "SeabedColumn(1 layer 10 m, 'sand' half-space, cp=1650 m/s, ρ=1.9513 g/cm³, α=0.8 dB/λ)",
    'Bottom':
        "Bottom('sand' half-space, cp=1650 m/s, ρ=1.9513 g/cm³, α=0.8 dB/λ)",
    'Bottom_rd':
        'Bottom(ranges [0, 2000, 5000] m, half-space)',
    'Surface':
        'Surface(vacuum)',
    'Thorp':
        'Thorp()',
    'FrancoisGarrison':
        'FrancoisGarrison(10 °C, 35 psu, pH 8)',
    'ConstantAbsorption':
        'ConstantAbsorption(0.01 dB/λ)',
    'BiologicalLayer':
        'BiologicalLayer(100–300 m, f0=5000 Hz, Q=10, a0=0.5 dB/km)',
    'Biological':
        'Biological(1 layer 100–300 m)',
    'AbsorptionCoefficient':
        'AbsorptionCoefficient(francois_garrison, 10 °C, 35 psu, pH 8, at 0 m, 12 frequencies 100–1000 Hz, dB/km)',
    'Environment':
        "Environment('shelf', depth=100 m, c=1500 m/s, seabed 'sand' half-space cp=1650 m/s, no absorption, ρw=1.027 g/cm³)",
    'Environment_full':
        "Environment('slope', depth=100–200 m, range-dependent, c=1500 m/s, seabed 'sand' half-space cp=1650 m/s, absorption=FrancoisGarrison, ρw=1.026 g/cm³)",
    'DataSource':
        'DataSource(id=gebco, name=GEBCO 2024 Grid, license=public domain)',
    'DataProvenance':
        'DataProvenance(source=gebco, data_point [43, 5], requested_point [43.01, 5.02], product=GEBCO_2024)',
    'Field':
        'Field(Bellhop, pressure Pa, frequency 200 Hz, 12 depths 5–95 m × 20 ranges 100–10000 m)',
    'Field_pinned':
        'Field(Kraken, level dB, frequency 200 Hz, 20 ranges 100–10000 m, at depth=50 m)',
    'Arrivals':
        'Arrivals(Bellhop, frequency 100 Hz, 2 arrivals, receiver depth 50 m, receiver range 1000 m)',
    'Modes':
        'Modes(Kraken, frequency 25 Hz, 2 modes, 12 depths 5–95 m)',
    'Rays':
        'Rays(Bellhop, frequency 100 Hz, 41 rays)',
    'Rays_eigen':
        'Rays(Bellhop, frequency 100 Hz, 1 eigenray, receiver depth 60 m, receiver range 3000 m)',
    'ReflectionCoefficient_broadband':
        'ReflectionCoefficient(OASR, frequencies [50, 100] Hz, angles [0, 45, 90] deg)',
    'GreensFunction':
        'GreensFunction(Scooter, frequency 100 Hz, receiver depths [50, 60] m, 256 phase speeds 1400–10000 m/s)',
    'Covariance':
        'Covariance(OASN, frequency 100 Hz, 8 receivers)',
    'Replicas':
        'Replicas(OASN, frequency 100 Hz, 10 depths 10–100 m, 8 receivers)',
    'ReflectionCoefficient':
        'ReflectionCoefficient(Bounce, frequency 50 Hz, 91 angles 0–90 deg)',
    'ResultStack':
        'ResultStack(2 Field slabs, source depths [10, 20] m)',
    'MediaTable':
        'MediaTable(water_density=1 g/cm³, tops 100 m, densities 1.9 g/cm³, bottom_depth=100 m, halfspace_density=1.9 g/cm³)',
    'SoundSpeeds':
        'SoundSpeeds(surface=1520 m/s, water_min=1495 m/s, water_max=1520 m/s, waveguide_min=1495 m/s, waveguide_max=1650 m/s)',
    'TimeSettings':
        'TimeSettings(pulse 128 samples, at 8000 Hz, record 1 s, t_start auto)',
    'WaveguideSpeeds':
        'WaveguideSpeeds(c=1495–1650 m/s)',
    'OutputSpec':
        'OutputSpec(Field pressure Pa coherent)',
    'Notice':
        "Notice('rays clipped', 'rays clipped at 80 deg', UACPYWarning)",
    'BandLevels':
        'BandLevels(5 centres 125–2000 Hz, 5 levels 70–80 dB re 1 µPa², band_type=octave, ref=1e-06)',
    'SpectralEstimate':
        'SpectralEstimate(257 frequencies 0–4000 Hz, 257 power 1 Pa²/Hz, scaling=density, method=welch)',
    'ProbabilisticSpectralEstimate':
        'ProbabilisticSpectralEstimate(257 frequencies 0–4000 Hz, 81 level_edges 40–120 dB re 1 µPa²/Hz, pdf 257×80 1/dB, ref=1e-06, scaling=density, method=welch)',
    'SpectrogramResult':
        'SpectrogramResult(129 frequencies 0–4000 Hz, 63 times 0–1 s, power 129×63 Pa²/Hz, scaling=density, mode=psd)',
    'CWTResult':
        'CWTResult(32 frequencies 100–4000 Hz, 64 times 0–0.063 s, coefficients 32×64 complex)',
    'WignerVilleResult':
        'WignerVilleResult(64 frequencies 0–4000 Hz, 64 times 0–0.063 s, distribution 64×64)',
    'Cepstrum':
        'Cepstrum(64 quefrencies 0–0.063 samples, 64 cepstrum 0)',
    'ComplexCepstrum':
        'ComplexCepstrum(64 cepstrum 0, delay=3 samples)',
    'CQTResult':
        'CQTResult(48 frequencies 100–4000 Hz, coefficients 48 complex Pa)',
    'CQSpectrogramResult':
        'CQSpectrogramResult(48 frequencies 100–4000 Hz, 64 times 0–0.063 s, power 48×64 Pa²/Hz, scaling=density)',
    'AmbiguityResult':
        'AmbiguityResult(201 delays_s -0.01–0.01 s, 101 doppler_hz -50–50 Hz, amplitude 101×201)',
    'FRFResult':
        'FRFResult(12 frequencies 100–1000 Hz, transfer_function 12 complex, method=h1)',
    'FKResult':
        'FKResult(65 frequencies 0–500 Hz, 32 wavenumbers -0.5–0.5 rad/m, power 65×32 Pa²/(Hz·rad/m), spectrum 65×32 complex, scaling=density)',
    'RadonResult':
        'RadonResult(21 moveout 0–0.001 s/m², 64 taus 0–0.063 s, panel 21×64, kind=parabolic)',
    'TauPResult':
        'TauPResult(21 slownesses 0–0.001 s/m, 64 taus 0–0.063 s, panel 21×64)',
    'BeamformResult':
        'BeamformResult(181 snr 0 dB, 181 angles -90–90 deg, peak_snr=12.5 dB)',
    'BeamformedField':
        'BeamformedField(response 181×12, 181 angles -90–90 deg, 12 element_power 1, 12 frequencies 100–1000 Hz)',
    'Snapshots':
        'Snapshots(frequency=500 Hz, data 8×16 complex)',
    'ChannelRegime':
        'ChannelRegime(coherence_bandwidth_hz=250 Hz, signal_bandwidth_hz=1000 Hz, rms_delay_spread_s=0.004 s, symbol_duration_s=0.001 s, frequency_selective=True, isi_symbols=4, convention=0.5)',
    'ChannelTaps':
        'ChannelTaps(taps 12 complex, 12 delays_s 0.66–0.6655 s, symbol_rate=2000 Hz, fc=12000 Hz, sps=4, first_arrival_s=0.66 s)',
    'BerCurve':
        'BerCurve(6 ebn0_dB 0–10 dB, 6 ber 1e-06–0.1, scheme=qpsk, n_bits=100000, n_train=400)',
    'JanusReception':
        'JanusReception(64 bits 0, crc_ok=True, doppler_scale=0)',
    'NoiseComponents':
        'NoiseComponents(12 total 1 dB re 1 µPa²/Hz, 12 wind 1 dB re 1 µPa²/Hz, 12 shipping 1 dB re 1 µPa²/Hz, 12 thermal 1 dB re 1 µPa²/Hz, 12 turbulence 1 dB re 1 µPa²/Hz)',
    'WenzNoise':
        'WenzNoise(12 frequencies 100–1000 Hz, wind=10 kn, depth=deep, shipping=medium, rain=no)',
    'BottomParameters':
        'BottomParameters(density_ratio=1.845, speed_ratio=1.178, loss_parameter=0.0122, volume_parameter=0.002, spectral_strength=0.0041)',
    'Bellhop':
        'Bellhop()',
    'Bellhop_knobs':
        "Bellhop(beam_type='B', n_beams=500)",
    'RAM':
        'RAM()',
    'AlongTrack':
        'AlongTrack(20 ranges 100–10000 m, 20 lats 43–43.1 °, 20 lons 5 °, 20 data 100–200, unit=m, quantity=depth, provenance=gebco)',
    'ArgoProfile':
        'ArgoProfile(platform=6902746, cycle=12, lat=43 °, lon=5 °, time=2024-05-01, 5 pressure_dbar 0–500 dbar, 5 temperature 13–20 °C, 5 salinity 38 psu, provenance=gebco)',
    'BathyGrid':
        'BathyGrid(11 lats 43–43.1 °, 21 lons 5–5.2 °, depths 11×21 m, provenance=gebco)',
    'Crust1Profile':
        'Crust1Profile(water_depth=2500 m, sediment_thickness=300 m, layer_names [upper sediments, middle sediments], thickness [100, 200] m, sound_speed [1600, 1800] m/s, shear_speed [0, 400] m/s, density [1.7, 1.9] g/cm³, provenance=gebco)',
    'PHProfile':
        'PHProfile(5 depths 0–500 m, 5 ph 7.9–8.1, ph_scale=total, provenance=gebco)',
    'SeabedSample':
        'SeabedSample(grain_size_phi=2.5, material=sand, folk_class=S, folk_class_scheme=folk, sample_point [43, 5], distance_km=3.2 km, provenance=gebco)',
    'TSProfile':
        'TSProfile(5 depths 0–500 m, 5 temperature 13–20 °C, 5 salinity 38 psu, temperature_kind=in-situ, provenance=gebco)',
    'SeaStateRecord':
        'SeaStateRecord(hs=1.5 m, tp=7 s, provenance=gebco)',
    'BoundaryTable':
        'BoundaryTable(kind=bathymetry, path=env.bty, interpolation=L, ranges [0, 5000] m, depths [100, 200] m)',
    'ShdFile':
        'ShdFile(title=shelf, frequencies 200 Hz, source_depths 20 m, 12 receiver_depths 5–95 m, 20 receiver_ranges 100–10000 m, pressure 1×1×1×12×20 complex)',
    'SspTable':
        'SspTable(path=env.ssp, ranges [0, 5000] m, sound_speed 3×2 m/s)',
    'PeGrid':
        'PeGrid(quantity=transmission_loss, unit=dB, 20 ranges 100–10000 m, 12 depths 5–95 m, data 12×20)',
    'ModelProvenance':
        'ModelProvenance(name=Acoustics Toolbox, authors=Michael B. Porter (HLS Research), license=GPL-3.0-or-later)',
    'ParallelResult':
        'ParallelResult(2 results, source_depth [10, 20])',
    'SonarBudget':
        'SonarBudget(mode=passive, source_level_dB=180 dB, detection_threshold_dB=10 dB, noise_level_dB=60 dB, directivity_index_dB=15 dB)',
    'JanusPacket':
        'JanusPacket(34 app_data 0)',
    'LinkResult':
        'LinkResult(ber=0.001, evm=0.12, scheme=qpsk, ebn0_dB=10 dB, tx_symbols 1000 complex, rx_symbols 1000 complex)',
    'ReceiverDiagnostics':
        'ReceiverDiagnostics(2000 bits 0, symbols 1000 complex, 4000 sync_metric 0, start=120)',
    'FRF':
        'FRF()',
    'ConvCode':
        'ConvCode()',
    'Modulator':
        'Modulator()',
    'DFE':
        'DFE()',
    'Transmitter':
        "Transmitter(modulation='qpsk', preamble=[64 values])",
    'CommsReceiver':
        "CommsReceiver(modulation='qpsk', preamble=[64 values])",
    'OFDMTransmitter':
        "OFDMTransmitter(modulation='qpsk')",
    'OFDMReceiver':
        "OFDMReceiver(modulation='qpsk')",
    'RamGrid':
        'RamGrid(frequency=200 Hz, dr=25.4846 m, dz=0.2589 m, zmax=321.812 m, n_depth_points=1244)',
    'KrakenLaunch':
        'KrakenLaunch(deck_frequency=200 Hz, 162 tabulation_depths 0–100 m, c_low=0 m/s, c_high 1732.5 m/s, rmax_m=10500 m, n_mesh=0, field_option=RC C)',
    'ModelSpec':
        'ModelSpec(8 modes, 5 supports, source_types [line, point])',
    'Surface_rd':
        'Surface(ranges [0, 5000] m, types [half-space, vacuum])',
    'Bounce':
        'Bounce()',
    'Kraken':
        'Kraken()',
    'Scooter':
        'Scooter()',
    'SPARC':
        'SPARC()',
    'OASN':
        'OASN()',
    'OASP':
        'OASP()',
    'OASR':
        'OASR()',
    'OASS':
        'OASS(correlation_length=10, rms_roughness=0.5)',
    'OASSP':
        'OASSP(correlation_length=10, rms_roughness=0.5)',
    'OAST':
        'OAST()',
    'Job':
        'Job(Bellhop, label=20)',
    'FlpFile':
        'FlpFile(title=shelf, option=RA, n_modes=12, profile_ranges 0 m, source_depths 20 m, 12 receiver_depths 5–95 m, 20 receiver_ranges 100–10000 m)',
    'Flp3dFile':
        'Flp3dFile(title=shelf, option=STD, n_modes=12, source_depths 20 m, 12 receiver_depths 5–95 m, 20 receiver_ranges 100–10000 m, 36 bearings 0–350 °)',
    'RtsFile':
        'RtsFile(title=shelf, positions 2×2, 1001 times 0–1 s, pressure 1001×2)',
    'Ssp3dFile':
        'Ssp3dFile(x [0, 500, 1000] m, y [0, 500, 1000] m, 12 z 5–95 m, sound_speed 12×3×3 m/s)',
    'PsifFile':
        'PsifFile(c0=1500 m/s, sample_rate=1024 Hz, q_factor=2, 12 frequencies 100–1000 Hz, 12 depths 5–95 m, 20 ranges 100–10000 m, pe_field 12×12×20 complex)',
    'OasesRhsHeader':
        'OasesRhsHeader(n_time_samples=1024, freq_min=50 Hz, freq_max=350 Hz, time_step=0.001 s, frequency=200 Hz, source_layer=2, n_wavenumbers=4096, interface=0)',
    'OasnNoise':
        'OasnNoise(surface_level=60 dB, white_level=40 dB)',
    'OasnReplicaGrid':
        'OasnReplicaGrid(z [10, 100, 10], c_low=1400 m/s, c_high=1e+08 m/s)',
}

#: Classes with no realistic instance of their own: abstract bases.
_ABSTRACT = {
    'uacpy.core.absorption.Absorption',
    'uacpy.core.results._base.Result',
    'uacpy.core.run_settings.EngineSettings',
    'uacpy.models.base.PropagationModel',
    'uacpy.models.oases._base.OASES',
}

#: A second constructor inside the parentheses: ``Name(`` after the head.
_NESTED_CONSTRUCTOR = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\(")


def _built():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return {key: build() for key, build in _instances().items()}


@pytest.fixture(scope='module')
def built():
    return _built()


def _public_classes():
    """Every class a public (no ``_``-prefixed component) uacpy module
    lists in ``__all__``, by dotted name; exceptions and enums excluded."""
    import enum
    found = {}
    for info in pkgutil.walk_packages(uacpy.__path__, 'uacpy.'):
        parts = info.name.split('.')[1:]
        if 'tests' in parts or 'third_party' in parts or any(
                p.startswith('_') for p in parts):
            continue
        module = importlib.import_module(info.name)
        for name in getattr(module, '__all__', ()):
            obj = getattr(module, name, None)
            if (inspect.isclass(obj) and obj.__module__.startswith('uacpy')
                    and not issubclass(obj, (BaseException, enum.Enum))):
                found[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return found


@pytest.mark.parametrize('key', sorted(EXPECTED))
def test_repr_is_the_pinned_one_line_string(built, key):
    assert repr(built[key]) == EXPECTED[key]


def test_every_public_class_prints_one_line_without_a_nested_constructor(built):
    bad = []
    for dotted, cls in sorted(_public_classes().items()):
        if dotted in _ABSTRACT:
            continue
        objs = [obj for obj in built.values() if type(obj) is cls]
        if not objs:
            bad.append(f"{dotted}: no instance in _instances()")
            continue
        for obj in objs:
            text = repr(obj)
            head = f"{cls.__name__}("
            if isinstance(obj, (RunSettings, EngineSettings)):
                ok = (text.startswith(head + '\n') and text.endswith('\n)'))
            else:
                inner = text[len(head):-1]
                ok = ('\n' not in text and text.startswith(head)
                      and text.endswith(')')
                      and not _NESTED_CONSTRUCTOR.search(inner))
            if not ok:
                bad.append(f"{dotted}: {text[:160]!r}")
    assert not bad, '\n'.join(bad)
