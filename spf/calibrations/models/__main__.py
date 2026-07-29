"""Export and query serial-specific phase-offset models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .export import export_model_registry
from .phase import load_model


def _export(args) -> int:
    registry = export_model_registry(
        matrix_path=args.matrix,
        output_root=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "models": len(registry["models"]),
                "radios": len(registry["radio_serials"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _predict(args) -> int:
    model = load_model(
        args.model,
        args.serial,
        registry_root=args.registry_root,
    )
    offset = model.predict_phase_offset(
        frequency_hz=args.frequency_hz,
        gain_rx1_db=args.gain_rx1_db,
        gain_rx2_db=args.gain_rx2_db,
        strict=not args.allow_unsupported_cell,
        allow_float32_frequency_alias=args.allow_float32_frequency_alias,
    )
    print(
        json.dumps(
            {
                "model": model.model_name,
                "serial": model.serial,
                "frequency_hz": args.frequency_hz,
                "gain_rx1_db": args.gain_rx1_db,
                "gain_rx2_db": args.gain_rx2_db,
                "phase_offset_rad": offset,
                "phase_offset_deg": offset * 180.0 / 3.141592653589793,
                "strict": not args.allow_unsupported_cell,
                "float32_frequency_alias_allowed": (args.allow_float32_frequency_alias),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export", help="export per-radio configs from a model-matrix JSON"
    )
    export_parser.add_argument("--matrix", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.set_defaults(function=_export)

    predict_parser = subparsers.add_parser(
        "predict", help="predict RX1-minus-RX2 phase offset"
    )
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--serial", required=True)
    predict_parser.add_argument("--frequency-hz", type=int, required=True)
    predict_parser.add_argument("--gain-rx1-db", type=int, required=True)
    predict_parser.add_argument("--gain-rx2-db", type=int, required=True)
    predict_parser.add_argument(
        "--registry-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    predict_parser.add_argument(
        "--allow-unsupported-cell",
        action="store_true",
        help="diagnostic only: bypass the strict passing-cell support gate",
    )
    predict_parser.add_argument(
        "--allow-float32-frequency-alias",
        action="store_true",
        help=(
            "recover an exact fitted LO from its integerized float32 "
            "representation; this does not enable interpolation"
        ),
    )
    predict_parser.set_defaults(function=_predict)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
