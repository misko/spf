# Cross-radio differential-phase interpretation

- Radio A: `104473b80a16000de6ff2000f8a6beca79`
- Radio B: `104000b299050013f4ff0700255e35222f`
- Common frequencies: 12
- Convention: radio_a intercept minus radio_b intercept

## Descriptive linear-delay fits

| Region | Frequencies | Effective delay | Free-space equivalent | Residual MAE / p95 / max |
|---|---:|---:|---:|---:|
| All frequencies | 12 | -44.31 ps | -13.28 mm | 14.77° / 44.13° / 46.19° |
| low full gain table | 3 | -30.38 ps | -9.11 mm | 0.15° / 0.22° / 0.22° |
| middle full gain table | 4 | 6.94 ps | 2.08 mm | 10.14° / 11.74° / 11.74° |
| high full gain table | 5 | -136.21 ps | -40.83 mm | 5.04° / 9.15° / 9.49° |
| vtx 5 7 to 5 9 ghz | 4 | -629.92 ps | -188.84 mm | 0.49° / 0.91° / 0.98° |

## Interpretation

A stable physical differential-path mismatch would produce a consistent delay across regions. Band-dependent sign or magnitude changes indicate that gain tables, analogue filtering, LO retunes, calibration state, or external cables also contribute.

Effective delays and free-space-equivalent lengths are descriptive slopes, not measurements of literal PCB trace length.
