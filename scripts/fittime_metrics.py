"""Shared metric selection for the fit-time aggregate/plot scripts.

Lets the whole fit-time toolchain switch between two timing notions held in the
modal_summary.json records:

  fit_time          whole adapter.fit() wall-clock (backbone load + feature
                    extraction + HP/Stage-1 (cached) + solve + XLA compile)
  train_fit_time    pure on-target training only (ADMM solve / finetune epoch
                    loop); excludes backbone load, cache reads, feature
                    extraction, HP selection  [added in BaseAdapter.train_time]

Each has a compile-excluded "warm" variant (mean over warm repeats; repeat 0 of
each K holds the JAX/XLA compile and is dropped):

  fit_time_warm     train_fit_time_warm

Select via the --metric CLI flag (added to each script) or the FITTIME_METRIC
env var. Default is fit_time_warm, so existing invocations and output filenames
are unchanged. train_* metrics write to train_-prefixed files so they never
clobber the fit-time figures/CSVs.

Usage:
    from fittime_metrics import resolve_metric, add_metric_arg
    metric = resolve_metric(args.metric)        # or resolve_metric()
    y = metric.get(rec)                          # None if the field is absent
    out = OUT_DIR / f"{metric.slug}fit_time_vs_k.png"
"""

from __future__ import annotations

import os

# name -> (ordered fallback field chain, axis label, output-filename slug)
_SPECS = {
    "fit_time": (["fit_time"], "Fit time (s)", ""),
    "fit_time_warm": (["fit_time_warm", "fit_time"], "Fit time (s, compile-excluded)", ""),
    "train_fit_time": (["train_fit_time"], "Train fit time (s)", "train_"),
    "train_fit_time_warm": (
        ["train_fit_time_warm", "train_fit_time"],
        "Train fit time (s, compile-excluded)",
        "train_",
    ),
}

CHOICES = list(_SPECS)


class Metric:
    """Resolved metric: how to read it from a record + how to label/name outputs."""

    def __init__(self, name: str):
        if name not in _SPECS:
            raise SystemExit(f"unknown --metric {name!r}; choose from {CHOICES}")
        self.name = name
        self.fallbacks, self.label, self.slug = _SPECS[name]

    def get(self, rec: dict):
        """Return the metric value from a per-(method,K) record, trying the
        fallback chain (e.g. train_fit_time_warm -> train_fit_time). Returns
        None when no field is present (e.g. old result files predating the
        train_fit_time metric) so callers can skip it."""
        for field in self.fallbacks:
            v = rec.get(field)
            if v is not None:
                return v
        return None

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Metric({self.name!r})"


def resolve_metric(cli_value: str | None = None, default: str = "fit_time_warm") -> Metric:
    """Resolve the metric from (in priority order) the --metric flag, the
    FITTIME_METRIC env var, then `default`."""
    name = cli_value or os.environ.get("FITTIME_METRIC") or default
    return Metric(name)


def add_metric_arg(parser, default: str = "fit_time_warm") -> None:
    """Register a standard --metric flag on an argparse parser."""
    parser.add_argument(
        "--metric",
        choices=CHOICES,
        default=None,
        help=f"timing metric to aggregate/plot (default: env FITTIME_METRIC or {default})",
    )
