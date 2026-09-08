"""Convenience entrypoint retained for the original one-file demo workflow.

The implementation now lives in ``anomaly_lab``. Running this file executes the
same leakage-aware benchmark exposed by the installed ``anomaly-lab`` CLI.
"""

from anomaly_lab.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["benchmark"]))
