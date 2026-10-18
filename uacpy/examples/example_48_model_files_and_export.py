"""Model files in, labelled data out — read a .shd and an .arr back, export.

Bellhop runs in a work directory the example names, so its files stay on disk:
the .shd of a coherent field and the .arr of an arrivals run. uacpy.io reads
them back into the Field and Arrivals the runs returned, and the results leave
uacpy as a NetCDF file, an xarray object and a pandas table — the route to
other tools. The export calls need the xarray extra
(pip install "uacpy[xarray]").

Uses: Bellhop(work_dir=) · result.metadata['shd_file'] / ['arr_file'] ·
uacpy.io.read_shd_file · uacpy.io.read_arr_file · Field.to_netcdf ·
Field.from_netcdf · Field.to_xarray · Arrivals.to_dataframe · Field.plot
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import uacpy

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

env = uacpy.Environment(
    name="Pekeris waveguide", bathymetry=100.0, ssp=1500.0,
    bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                    sound_speed=1600.0, density=1.5,
                                    attenuation=0.5))
source = uacpy.Source(depths=50.0, frequencies=100.0)
receiver = uacpy.Receiver(depths=np.linspace(5, 95, 37),   # 2.5 m: 30 m is one
                          ranges=np.linspace(100, 5000, 200))

# A pinned work_dir keeps the engine's files after the run (cleanup defaults
# to False there); a scratch directory under OUT holds both runs.
scratch = tempfile.TemporaryDirectory(dir=OUT)
work = Path(scratch.name)
bellhop = uacpy.Bellhop(backend='fortran', work_dir=work / 'field')
field = bellhop.run(env, source, receiver)
arrivals = bellhop.copy(work_dir=work / 'arrivals').run(
    env, source, uacpy.Receiver(depths=[30.0], ranges=[2000.0]),
    run_mode=uacpy.RunMode.ARRIVALS)

# The result names the file it was read from. The reader returns the
# engine's own normalisation: the level is the run's, the sign is Bellhop's
# (the model layer flips it onto the package convention after reading).
shd = uacpy.io.read_shd_file(field.metadata['shd_file'])
arr = uacpy.io.read_arr_file(arrivals.metadata['arr_file'])
print(f"  .shd: {shd.data.shape} complex pressure, max |ΔTL| against the "
      f"run {np.nanmax(np.abs(shd.dB - field.dB)):.1e} dB, phase offset "
      f"{np.nanmedian(np.angle(shd.data / field.data, deg=True)):+.0f}°")
print(f"  .arr: {arr.n_arrivals} arrivals, the run's {arrivals.n_arrivals}")

# Out of uacpy: a NetCDF file any reader opens (complex data as real and
# imaginary parts), read back exactly; an xarray DataArray with every axis
# and its unit; one row per arrival as a pandas table, written as CSV.
nc_path = work / 'field.nc'
field.to_netcdf(nc_path)
back = uacpy.Field.from_netcdf(nc_path)
labelled = field.to_xarray()
table = arrivals.to_dataframe()
table.to_csv(work / 'arrivals.csv', index=False)
print(f"  NetCDF: {nc_path.stat().st_size / 1e3:.0f} kB, read back "
      f"{'exactly' if np.array_equal(back.data, field.data) else 'CHANGED'}")
print(f"  xarray: dims {labelled.dims}, units {labelled.attrs['units']!r}")
print(f"  pandas: {table.shape[0]} rows x {table.shape[1]} columns, "
      f"strongest path {table.loc[table.amplitude.idxmax(), 'kind']!r}, "
      f"first arrival at {table.delay.min():.4f} s")

# The three copies of the field drawn as one TL cut: run, .shd, NetCDF.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
field.plot(env=env, ax=axes[0], title='Bellhop run (100 Hz)')
copies = ((field, 'run', dict(lw=5, alpha=0.35)),
          (shd, '.shd read back', dict(ls='--')),
          (back, 'NetCDF read back', dict(ls=':', color='k')))
for copy, label, style in copies:
    copy.at(depth=30.0).plot(ax=axes[1], label=label,
                             title='TL at 30 m — three copies, one curve',
                             **style)
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT / 'example_48_model_files_and_export.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
scratch.cleanup()
print(f"wrote {OUT / 'example_48_model_files_and_export.png'}")
