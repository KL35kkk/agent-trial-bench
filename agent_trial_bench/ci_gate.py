"""
CI/CD Gate – automated deployment decision system.

Usage as a Python API::

    from agent_trial_bench.ci_gate import CIGate, GateConfig

    gate = CIGate(GateConfig(min_success_rate=0.92, max_cost_increase_pct=5.0))
    result = gate.evaluate(current_results, baseline_results)
    if not result.passed:
        print(result.report)
        sys.exit(1)

Usage as a CLI::

    python -m agent_trial_bench.ci_gate \\
        --current  output/results.json \\
        --baseline output/baseline.json \\
        --min-success-rate 0.92 \\
        --max-cost-increase 5.0

Exit code 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .analyzer.regression_detector import (
    RegressionDetector,
    RegressionReport,
    TaskSnapshot,
)

logger = logging.getLogger(__name__)


# ── configuration ─────────────────────────────────────────────────────────────


@dataclass
class GateConfig:
    """Rules that the CI gate checks against.

    Attributes:
        min_success_rate:        Minimum fraction of tasks that must pass
                                 (default: 0.92 = 92 %).
        max_cost_increase_pct:   Maximum allowed cost increase relative to
                                 baseline (default: 5.0 %).  Set to None to
                                 skip cost check.
        max_new_failure_modes:   Maximum number of new failure categories
                                 allowed (default: 0).
        max_regressions:         Maximum number of regressed tasks allowed
                                 (default: 0).
        max_variance:            If set, tasks with variance above this value
                                 are flagged but do not block by default.
        unstable_task_limit:     Maximum number of UNSTABLE tasks before
                                 blocking (default: None = unlimited).
        regression_threshold:    Score drop that counts as a regression
                                 (passed to RegressionDetector).
    """

    min_success_rate: float = 0.92
    max_cost_increase_pct: Optional[float] = 5.0
    max_new_failure_modes: int = 0
    max_regressions: int = 0
    max_variance: Optional[float] = None
    unstable_task_limit: Optional[int] = None
    regression_threshold: float = 0.05


# ── result ────────────────────────────────────────────────────────────────────


@dataclass
class GateResult:
    """Output of a CI Gate evaluation."""

    passed: bool
    blocking_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    regression_report: Optional[RegressionReport] = None
    report: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "blocking_issues": self.blocking_issues,
            "warnings": self.warnings,
            "metrics": self.metrics,
            "regression_report": self.regression_report.to_dict() if self.regression_report else None,
        }


# ── gate ──────────────────────────────────────────────────────────────────────


class CIGate:
    """Evaluates evaluation results against a set of quality rules.

    Args:
        config: Gate configuration.  Uses defaults if not provided.
    """

    def __init__(self, config: Optional[GateConfig] = None) -> None:
        self.config = config or GateConfig()

    # ── public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        current_results: Sequence[Any],
        baseline_results: Optional[Sequence[Any]] = None,
    ) -> GateResult:
        """Run all gate checks and return a GateResult.

        Args:
            current_results: Results from the run under test.  Each item must
                be a dict or object with ``task_id`` and a score field
                (``mean_score`` or ``evaluation_score``).
            baseline_results: Results from the reference run.  If None, only
                absolute checks (success_rate, cost) are performed.
        """
        issues: List[str] = []
        warnings: List[str] = []
        metrics: Dict[str, Any] = {}

        current_snaps = RegressionDetector.snapshots_from_results(current_results)
        baseline_snaps = (
            RegressionDetector.snapshots_from_results(baseline_results)
            if baseline_results
            else None
        )

        # ── 1. Success rate ───────────────────────────────────────────────────
        success_rate = _compute_success_rate(current_results)
        metrics["success_rate"] = success_rate
        if success_rate < self.config.min_success_rate:
            issues.append(
                f"Success rate {success_rate:.1%} < required {self.config.min_success_rate:.1%}"
            )

        # ── 2. Cost check ─────────────────────────────────────────────────────
        if self.config.max_cost_increase_pct is not None and baseline_snaps:
            cost_delta_pct = _compute_cost_delta_pct(current_results, baseline_results)
            metrics["cost_delta_pct"] = cost_delta_pct
            if cost_delta_pct is not None and cost_delta_pct > self.config.max_cost_increase_pct:
                issues.append(
                    f"Cost increase {cost_delta_pct:.1f}% > allowed {self.config.max_cost_increase_pct:.1f}%"
                )

        # ── 3. Regression detection ───────────────────────────────────────────
        reg_report: Optional[RegressionReport] = None
        if baseline_snaps:
            detector = RegressionDetector(regression_threshold=self.config.regression_threshold)
            reg_report = detector.compare(baseline_snaps, current_snaps)
            metrics["regression_count"] = reg_report.regression_count
            metrics["new_failure_modes"] = reg_report.new_failure_modes
            metrics["score_delta"] = reg_report.score_delta

            if reg_report.regression_count > self.config.max_regressions:
                issues.append(
                    f"{reg_report.regression_count} regression(s) detected "
                    f"(limit: {self.config.max_regressions}). "
                    f"Regressed tasks: {[r['task_id'] for r in reg_report.regressions[:5]]}"
                )

            if len(reg_report.new_failure_modes) > self.config.max_new_failure_modes:
                issues.append(
                    f"{len(reg_report.new_failure_modes)} new failure mode(s) detected: "
                    f"{reg_report.new_failure_modes}"
                )

        # ── 4. Stability / variance check ─────────────────────────────────────
        if self.config.max_variance is not None or self.config.unstable_task_limit is not None:
            unstable = _count_unstable(current_results)
            metrics["unstable_tasks"] = unstable
            if self.config.unstable_task_limit is not None and unstable > self.config.unstable_task_limit:
                issues.append(
                    f"{unstable} UNSTABLE task(s) (limit: {self.config.unstable_task_limit})"
                )
            elif unstable > 0:
                warnings.append(
                    f"{unstable} task(s) marked UNSTABLE (high variance). Manual review recommended."
                )

        passed = len(issues) == 0
        report = _build_report(passed, issues, warnings, metrics, reg_report)

        return GateResult(
            passed=passed,
            blocking_issues=issues,
            warnings=warnings,
            metrics=metrics,
            regression_report=reg_report,
            report=report,
        )


# ── helpers ───────────────────────────────────────────────────────────────────


def _compute_success_rate(results: Sequence[Any]) -> float:
    if not results:
        return 0.0
    total = len(results)
    passed = 0
    for r in results:
        if isinstance(r, dict):
            score = float(r.get("mean_score") or r.get("evaluation_score") or 0.0)
        else:
            score = float(getattr(r, "mean_score", None) or getattr(r, "evaluation_score", 0.0))
        if score >= 0.7:  # standard pass threshold
            passed += 1
    return passed / total


def _compute_cost_delta_pct(
    current: Sequence[Any],
    baseline: Optional[Sequence[Any]],
) -> Optional[float]:
    if not baseline:
        return None
    cur_cost = _sum_cost(current)
    base_cost = _sum_cost(baseline)
    if base_cost == 0:
        return None
    return round((cur_cost - base_cost) / base_cost * 100, 2)


def _sum_cost(results: Sequence[Any]) -> float:
    total = 0.0
    for r in results:
        if isinstance(r, dict):
            total += float(r.get("cost_usd") or r.get("aggregated_metrics", {}).get("total_cost_usd", 0) or 0)
        else:
            m = getattr(r, "aggregated_metrics", {}) or {}
            total += float(m.get("total_cost_usd", 0) or 0)
    return total


def _count_unstable(results: Sequence[Any]) -> int:
    count = 0
    for r in results:
        if isinstance(r, dict):
            if r.get("is_unstable"):
                count += 1
        else:
            if getattr(r, "is_unstable", False):
                count += 1
    return count


def _build_report(
    passed: bool,
    issues: List[str],
    warnings: List[str],
    metrics: Dict[str, Any],
    reg_report: Optional[RegressionReport],
) -> str:
    lines: List[str] = [
        "=" * 60,
        f"CI GATE: {'✅ PASS' if passed else '❌ FAIL'}",
        "=" * 60,
        "",
        "Metrics:",
    ]
    for k, v in metrics.items():
        lines.append(f"  {k:<30} {v}")

    if issues:
        lines.append("")
        lines.append("Blocking issues:")
        for issue in issues:
            lines.append(f"  ❌ {issue}")

    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in warnings:
            lines.append(f"  ⚠️  {w}")

    if reg_report and (reg_report.regressions or reg_report.improvements):
        lines.append("")
        lines.append(f"Score delta: {reg_report.score_delta:+.4f}")
        if reg_report.regressions:
            lines.append(f"Regressions ({reg_report.regression_count}):")
            for r in reg_report.regressions[:5]:
                lines.append(
                    f"  {r['task_id']}: {r['baseline_score']:.3f} → {r['candidate_score']:.3f}"
                    f" ({r['delta']:+.3f})"
                )

    lines.append("=" * 60)
    return "\n".join(lines)


# ── CLI entry point ───────────────────────────────────────────────────────────


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m agent_trial_bench.ci_gate",
        description="DAB Evaluation CI/CD Gate – checks evaluation results against quality rules.",
    )
    parser.add_argument("--current", required=True, help="Path to current evaluation results (JSON).")
    parser.add_argument("--baseline", default=None, help="Path to baseline results (JSON).  Optional.")
    parser.add_argument("--min-success-rate", type=float, default=0.92, help="Minimum success rate (default: 0.92).")
    parser.add_argument("--max-cost-increase", type=float, default=5.0, help="Max cost increase %% (default: 5.0).")
    parser.add_argument("--max-regressions", type=int, default=0, help="Max regressions allowed (default: 0).")
    parser.add_argument("--max-new-failure-modes", type=int, default=0, help="Max new failure modes (default: 0).")
    parser.add_argument("--output", default=None, help="Write gate result JSON to this path.")
    args = parser.parse_args()

    current_path = Path(args.current)
    if not current_path.exists():
        print(f"Error: --current file not found: {current_path}", file=sys.stderr)
        sys.exit(2)

    current_results = json.loads(current_path.read_text(encoding="utf-8"))
    baseline_results = None
    if args.baseline:
        baseline_path = Path(args.baseline)
        if baseline_path.exists():
            baseline_results = json.loads(baseline_path.read_text(encoding="utf-8"))
        else:
            print(f"Warning: --baseline file not found: {baseline_path}. Skipping regression check.", file=sys.stderr)

    config = GateConfig(
        min_success_rate=args.min_success_rate,
        max_cost_increase_pct=args.max_cost_increase,
        max_regressions=args.max_regressions,
        max_new_failure_modes=args.max_new_failure_modes,
    )

    gate = CIGate(config)
    result = gate.evaluate(current_results, baseline_results)

    print(result.report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nGate result saved to {out_path}")

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    _cli()
