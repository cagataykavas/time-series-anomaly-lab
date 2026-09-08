from __future__ import annotations

import argparse
import json
from pathlib import Path

from anomaly_lab.calibration import QuantileCalibrator, RobustSigmaCalibrator
from anomaly_lab.experiment import run_benchmark
from anomaly_lab.synthetic import generate_synthetic_series


def _build_calibrator(args: argparse.Namespace):
    if args.calibration == "quantile":
        return QuantileCalibrator(args.quantile)
    return RobustSigmaCalibrator(args.sigma)


def command_benchmark(args: argparse.Namespace) -> int:
    dataset = generate_synthetic_series(
        n=args.samples,
        seed=args.seed,
        calibration_fraction=args.calibration_fraction,
    )
    report = run_benchmark(
        dataset,
        calibrator=_build_calibrator(args),
        fit_fraction_of_calibration=args.fit_fraction,
        event_tolerance=args.event_tolerance,
    )
    payload = report.to_dict()
    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anomaly-lab",
        description="Run leakage-aware time-series anomaly detection benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--samples", type=int, default=5000)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--calibration-fraction", type=float, default=0.35)
    benchmark.add_argument("--fit-fraction", type=float, default=0.65)
    benchmark.add_argument("--event-tolerance", type=int, default=3)
    benchmark.add_argument(
        "--calibration",
        choices=("quantile", "robust-sigma"),
        default="quantile",
    )
    benchmark.add_argument("--quantile", type=float, default=0.995)
    benchmark.add_argument("--sigma", type=float, default=4.0)
    benchmark.add_argument("--output")
    benchmark.set_defaults(handler=command_benchmark)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
