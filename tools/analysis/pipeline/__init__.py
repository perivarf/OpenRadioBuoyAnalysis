"""The wave processing pipeline, one module per step of the generic pipeline in the
thesis (method/met_processing_pipeline.tex, table tab:pipeline-steps):

  step   module          in -> out
  1-2    session_input   path            -> Session        (logged rows, GPS, cfg)
  3      raw_stream      Session         -> RawSeries      (AHRS on the raw log)
  4      ahrs            Session         -> VerticalAccel  (per method, row rate)
  5      decimation      VerticalAccel   -> AnalysisSeries (analysis rate, masks)
  6-8    spectral        AnalysisSeries  -> Spectra        (detrend, Welch)
  9      elevation       Spectra         -> ElevationSpectra (÷ω⁴, ÷ω²)
  10     cutoff          ElevationSpectra -> LowFreqCut    (taper / f_c)
  11     sea_state       ElevationSpectra + LowFreqCut -> SeaState (Hs, Tp, Tz, Tc)
  12     output, report  everything      -> files, console

settings holds one dataclass per step, collected in Settings. postprocess.py
chains the steps: analyse_session(path, st) -> Result.
"""
