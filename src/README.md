# WASP-39b JWST PRISM pipeline (refactor)

## Layout
- `src/wasp39/io.py` – HDF5 I/O and saving the spectrum table
- `src/wasp39/normalize.py` – white-light and OOT normalization
- `src/wasp39/binning.py` – time binning + wavelength bin helpers
- `src/wasp39/lightcurve.py` – BATMAN transit model wrapper
- `src/wasp39/mcmc.py` – emcee helpers (bin-depth fits)
- `src/wasp39/spectrum.py` – transmission spectrum construction
- `src/wasp39/plotting.py` – plotting utilities
- `src/wasp39/main.py` – pipeline entrypoint
- `src/run.py` – convenience runner

## Run
From the folder that contains `src/`:
```bash
python -m wasp39.main
```

If your environment doesn’t automatically include `src/` on `PYTHONPATH`, either:
```bash
PYTHONPATH=src python -m wasp39.main
```
or run:
```bash
python src/run.py
```

## Requirements
`batman-package`, `emcee`, `corner`, `h5py`, `numpy`, `matplotlib`
