# E-GSC8 standard-IIO USB report

The result and interpretation are maintained in
[`experiments/e_gsc8_carrier_transfer_5840/RESULTS.md`](../../../../../experiments/e_gsc8_carrier_transfer_5840/RESULTS.md).
This directory contains the machine-readable primary-session grading output.

- Firmware: `v0.38-plutoplus-spf-libiio-metadata-v5`, persistent QSPI
- Host libiio: SPF 0.25 with `MetadataBuffer`
- Capture: 816/816 valid frames; 272/272 passing cells
- Outcome: H3 passes on both; H1 passes on clean R18 and conservatively fails
  on damaged R17; H2 passes on both
- Raw data:
  `/mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/e_gsc8_iio_usb_20260813_v1/`
