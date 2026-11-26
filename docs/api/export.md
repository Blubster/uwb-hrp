# ARB Waveform Export

Vendor-neutral active-power normalization, semantic markers, and the built-in Rohde & Schwarz `.wv` exporter.

## Vendor-Neutral Core

::: uwb_hrp.export.base
    options:
      show_submodules: false
      members: false

::: uwb_hrp.export.ARBExporter

::: uwb_hrp.export.WaveformPowerMetrics

::: uwb_hrp.export.MarkerTrack

::: uwb_hrp.export.WaveformMarkers

::: uwb_hrp.export.PreparedARBWaveform

::: uwb_hrp.export.compute_power_metrics

::: uwb_hrp.export.build_waveform_markers

::: uwb_hrp.export.prepare_arb_waveform

## Rohde & Schwarz `.wv` Backend

::: uwb_hrp.RSWaveformExporter

::: uwb_hrp.save_rs_waveform

::: uwb_hrp.to_rs_waveform

::: uwb_hrp.export.build_rs_markers

::: uwb_hrp.export.compute_rs_power_metrics
