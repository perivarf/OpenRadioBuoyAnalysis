# tools/ – analysis of the ORB drifter logs

Python tools that turn the buoy's logs (`<stamp>_raw.bin`, `_imu.csv`, `_gps.csv`,
`_ses.csv`, `_cfg.csv`) into wave spectra, wave parameters (Hs, Tp, Tz, Tc) and the
figures and tables used in the thesis. The measurement directories next to `tools/`
hold the data; each has a `plot.sh` (and sometimes `test_config.sh`) that runs the
figure chain for that measurement.

## Setup

- Python 3 with `numpy`, `pandas`, `matplotlib`, `Pillow` (map cropping/tiles) and
  `scipy` (one optional fit).
- Optional: `ahrs_fusion`, the compiled NXP Sensor Fusion module the SFY buoy runs,
  for the `nxp` filter (`filters/kalman_nxp.py`). Build it from the ahrs-fusion repo
  with `maturin develop --features python`. Everything else works without it.
- Network for map tiles (OpenStreetMap, cached in `~/.cache/orb_mapplot`) and for
  the model comparisons (thredds.met.no, kystvarslingssenteret.no; cached in
  `~/.cache/orb_met`). PSDs are cached in `~/.cache/orb_psd`.

### How imports work

The entry points are the `script_*.py` files directly in `tools/`. Each one does
`import _paths` first; `_paths.py` puts `tools/` and every library subdirectory on
`sys.path`, so modules import each other by bare name (`import postprocess`,
`from osm import ...`), whatever folder they are in. `pyrightconfig.json` gives the
editor the same paths. The only package is `analysis/pipeline/` (`from pipeline
import ...`, relative imports inside).

Run the scripts from the repo root (the folder above `tools/`); the session sets
resolve their directories from there:

    python3 tools/script_plot.py SKJAERHALDEN

## Typical flow

```
<stamp>_raw.bin ──► conversion/ (rawlog, raw_to_imu_csv)
                     │  reconstructed imu.csv: FIR decimation + AHRS, as on board
                     ▼
               analysis/postprocess.py  (pipeline steps 1-12)
                     │  <stamp>_ana_python.csv, <stamp>_spec_python.csv
                     ▼
   script_plot.py ──► map + table per session, overview of the series
   script_felles_psd / script_hs_cutoff / script_psd_comp ──► comparison figures
   script_met_waves / script_norkyst160 ──► against models (KVS, NorKyst)
```

`plot.sh` in a measurement directory sets a few environment variables (the FIR
taps, raw averaging, brake rejection) and runs, for the set it names:
`script_plot.py` → `script_felles_psd.py` → `script_hs_cutoff.py` →
`script_psd_comp.py` → `script_met_waves.py`. With `PLOT_ONLY=1` every tool only
redraws from the saved `data/` CSVs. `test_config.sh` runs `script_test_config.py`.

## Scripts (entry points)

Each script has a docstring with more examples; `--help` lists every flag.

| Script | What it does | Writes |
|---|---|---|
| `script_plot.py SET` | The main figure chain: postprocess on every session in a set with the same taper, then a map per session and an overview map + table. Sessions with only a raw log are reconstructed first. | `<stamp>_map.png`, `<stamp>_ana_python.csv`, `_spec_python.csv` per session; `<prefix>_overview/map/table.png` |
| `script_postprocess.py PATH` | Analysis of one session (a session folder or `*_imu.csv`) and a console report. | `<stamp>_ana_python.csv`, `<stamp>_spec_python.csv`, `<stamp>_spec_acc/eta.png` |
| `script_mapplot.py DIR...` | Map figure(s) for individual session folders, outside a set. | `<stamp>_map.png` (or `-o`) |
| `script_felles_psd.py SET` | All sessions of a series in one elevation and one acceleration spectrum, per taper. | `taper_sweep/<prefix>_eta_<method>_<f1>_<f2>.png`, `_acc_<method>.png` (+ `data/`) |
| `script_hs_cutoff.py SET` | Hs as a function of the low-frequency cutoff, one curve per session. | `taper_sweep/<prefix>_hs_vs_fc_<method>.png` (+ `data/`) |
| `script_psd_comp.py SET` | Untapered PSDs from Madgwick and Kalman against GPS, and their ratios. `--report` adds a summary over the series. | `<stamp>_psd_utapert.png`, `_psd_forhold.png`, `<prefix>_psd_rapport.png` |
| `script_met_waves.py SET` | Modelled Hs from Kystvarslingssenteret (KVS) at each session, next to the buoy's own Hs. Reads existing `_ana_python.csv`. | `<config>/<prefix>_met_waves_<config>.csv`, `<stamp>_met_waves_<config>.png` |
| `script_norkyst160.py [DIR]` | NorKyst v3 surface current, wind and sea level at each session, next to the buoy's drift. | console, `--csv` |
| `script_test_config.py SET --session S` | Hs for one session under many processing combinations (raw averaging, FIR, Welch segment, capture length, Welch rate). Called from `test_config.sh`. | `test_config/<prefix>_test_config*.csv/png` |
| `script_filter_timing.py SET` | CPU time per orientation filter (Madgwick, Kalman, NXP) for every session. | `<prefix>_filter_timing.csv` |
| `script_rawlog.py FILE` | Summary of a `_raw.bin`: header, rates, lost blocks. | console |
| `script_lora_range.py` | Map of the LoRa range test: base station and the drifter positions where a message got through, with distances. The points are constants in the script (or `--csv`). | `lora_range/lora_range.png` in the repo root |
| `script_raw_to_imu_csv.py PATH` | `_raw.bin` → raw sample CSV (`--mode samples`) or a reconstructed device-format `imu.csv` (`--mode imu`, `--compare` against the device's own). | CSV next to the `.bin` (or `-o`) |
| `script_power_log.py` | Logs voltage/current from the GPP-4323 (needs the instrument). | `<prefix>_raw.csv`, `<prefix>_sum.csv` |
| `script_power_batch_log.py --csv P` | Logs a whole test batch (firmware `power_batch.cpp`), follows the markers live, then analyzes it. `--replay` replays a log without the instrument. | `<prefix>_raw.csv`, `_tests.csv`, `.png` |
| `script_power_batch_analyze.py RAW` | Finds the tests in a batch log from the current markers. | `<prefix>_tests.csv` (mean/std/energy per test) |
| `script_power_batch_results.py` | Current per test over the analysis window and the component table (differences between tests). | `power_measurement/batch_results/` |
| `script_power_baseline_k.py` | As above with the deep-sleep floor from the Keithley reading. **The thesis version** (`energy_overview.png`, table `tab:power_components`). | `power_measurement/batch_results_baseline_k/` |
| `script_power_keithley.py` | Variant with every test corrected against the Keithley readings. | `power_measurement/batch_results_keithley/` |
| `script_power_overview.py` | Overview of the single (non-batch) measurements. | `oversikt_plot.png`, `komponent_tabell.*` |
| `script_power_plot.py --csv P` | Current and power for one log. | `<prefix>_plot.png` |

`SET` is a set name from `analysis/session_sets.py` (case-insensitive) or a
measurement directory without a set (it is scanned for session folders).
`--config NAME` puts the output in `<measurement>/NAME/` and adds `_NAME` to the
file names (see `analysis/configs.py`).

## Library folders

### `analysis/`
| Module | Contents |
|---|---|
| `postprocess.py` | Chains the pipeline steps: `analyse_session(path, settings)` and `process()` (with files and report). |
| `pipeline/` | One module per processing step, see the table below. `settings.py` holds one dataclass per step, gathered in `Settings`. |
| `session_sets.py` | The named session sets: directory, sessions in order, labels, styles, file prefix, taper. Add a measurement series here. |
| `session_source.py` | Chooses raw log over device `imu.csv`, and builds the work directory with a reconstructed `imu.csv` (`ensure_imu`). Reads `RAW_AVERAGE`. |
| `configs.py` | Config names → output directories and file-name suffixes. |
| `config_test.py` | The combinations behind `script_test_config.py`. |
| `filter_timing.py` | The timing behind `script_filter_timing.py`. |

Pipeline steps (`analysis/pipeline/`, as in the thesis' pipeline table):

| Step | Module | In → out |
|---|---|---|
| 1-2 | `session_input` | path → Session (logged rows, GPS, cfg) |
| 3 | `raw_stream` | Session → RawSeries (AHRS on the raw log) |
| 4 | `ahrs` | Session → VerticalAccel per method. `METHODS` lists the filters; add one there and every table and figure follows. |
| 5 | `decimation` | → AnalysisSeries (analysis rate, gap/overflow masks) |
| 6-8 | `spectral` | → Spectra (detrend, Welch) |
| 9 | `elevation` | → ElevationSpectra (÷ω⁴ for accel, ÷ω² for GPS velocity) |
| 10 | `cutoff` | → LowFreqCut (taper or f_c, GPS reference) |
| 11 | `sea_state` | → SeaState (Hs, Tp, Tz, Tc) |
| 12 | `output`, `report` | CSV files, spectrum figures, console report |

### `filters/`
Orientation filters with a common interface (`init_from_accel`, `update(gx, gy, gz,
ax, ay, az, dt)`, `.q`): `madgwick.py` (identical to the firmware), `kalman.py`
(Kok et al. 2017 EKF, ported to the firmware), `kalman_nxp.py` (NXP, needs
`ahrs_fusion`). Also `rotation.py` (quaternion maths) and `fir.py` (FIR lowpass +
decimation, as the SFY buoy).

### `psd/`
`welch.py` (Welch PSD with segment rejection, as `analysis.cpp`) and `spectra.py`
(the cached spectrum chain shared by `felles_psd`, `hs_cutoff` and `psd_comp`).

### `plots/`
`mapplot.py` (maps, tables, overview), `felles_psd.py`, `hs_cutoff.py`,
`psd_comp.py`, plus helpers `freq_axis.py` (the shared log-Hz axis) and `trim.py`
(even margins on saved PNGs).

### `conversion/`
`rawlog.py` reads the `_raw.bin` format (documented in its docstring);
`raw_to_imu_csv.py` rebuilds the device's `imu.csv` from it.

### `models/`, `norce_kystverket/`, `maps/`
- `models/norkyst160.py`: NorKyst v3 via OPeNDAP; `models/opendap.py`: shared
  OPeNDAP access; `models/session_points.py`: time, position and drift per session.
- `norce_kystverket/kystvarsling.py`: KVS wave forecasts and the comparison tables.
- `maps/osm.py`: OSM tiles, Web Mercator, scale bar and north arrow (also used by
  `script_lora_range.py`).

### `power/` – power consumption bench
The power measurements of ORB v2 (GW Instek GPP-4323 power supply over LAN,
Keithley 2000 as reference). The `script_power_*.py` scripts read and write
`power_measurement/` in the repo root by default (`power_io.DATA_DIR`).

| Module | Contents |
|---|---|
| `gpp4323.py` | The GPP-4323 over SCPI, and sample sources: `sample()` from the instrument, `replay()` from a `_raw.csv`. |
| `power_io.py` | `_raw.csv` / `_sum.csv` / `legend_names.csv` reading and writing, prefixes. |
| `power_stats.py` | Time-weighted mean/std, integrals, boxcar smoothing, window statistics. |
| `batch_markers.py` | The test IDs and batches of the firmware (`power_batch.cpp`), and detection of the ID and capture-start markers in the current. |
| `batch_analysis.py` | Splits a batch log into tests and measures each one (`analyze_log` → `<prefix>_tests.csv`). |
| `batch_live.py` | `BatchTracker`: follows a batch while it is logged (new tests, SD-card swaps, when to stop). |
| `batch_components.py` | Test labels, the components as differences between tests, the component table (CSV/TeX/PNG), the Keithley fit. |
| `single_components.py` | The component table for the older single measurements. |
| `power_plots.py` | Current-over-time figure, batch log with markers, single log. |

### `cli/`
`arguments.py`: the command-line flags shared by the scripts. Defaults come from
the settings dataclasses, so each default is defined once.


## Output files per session

| File | Contents |
|---|---|
| `<stamp>_ana_python.csv` | Wave parameters per method (`Hs_<method>`, Tp, Tz, Tc), taper, settings |
| `<stamp>_spec_python.csv` | The spectra per method (acceleration and elevation) |
| `<stamp>_spec_acc.png`, `_spec_eta.png` | The spectra as figures |
| `<stamp>_map.png` | Map, track and fact table |
| `data/*.csv` | The numbers each figure is drawn from (`--plot-only` redraws from them) |

Work files that can be regenerated (`*_raw/` directories, `*_imu_raw_*.csv`, etc.)
are in `.gitignore`.
