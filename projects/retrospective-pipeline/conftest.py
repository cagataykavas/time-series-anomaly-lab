"""Keep this optional neural evidence suite separate from core-only CI."""
import importlib.util

# Core repository CI deliberately installs only the non-neural package.
# The reproduction command installs requirements.txt and runs all 24 tests.
collect_ignore = (
    ["test_pipeline.py"]
    if any(
        importlib.util.find_spec(name) is None
        for name in ("torch", "numpy", "matplotlib", "plotly", "pandas")
    )
    else []
)
