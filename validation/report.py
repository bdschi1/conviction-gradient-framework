"""Validation report formatting.

Formats comparison tables and ablation summaries for console output.
"""

from __future__ import annotations

import numpy as np

from validation.runner import ValidationResult


def format_comparison_table(result: ValidationResult) -> str:
    """Format a comparison table of all strategies.

    Args:
        result: Single-seed validation result.

    Returns:
        Formatted string table.
    """
    lines = []
    header = (
        f"{'Strategy':<25} {'Ann Ret':>8} {'Ann Vol':>8} "
        f"{'Sharpe':>8} {'MaxDD':>8} {'Turnover':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    # Baselines first
    for b in result.baselines:
        m = b.metrics
        lines.append(
            f"{b.name:<25} {m.annualized_return:>8.3f} {m.annualized_vol:>8.3f} "
            f"{m.sharpe_ratio:>8.3f} {m.max_drawdown:>8.3f} {m.turnover:>10.3f}"
        )

    lines.append("-" * len(header))

    # CGF variants
    for v in result.variants:
        m = v.metrics
        lines.append(
            f"{v.name:<25} {m.annualized_return:>8.3f} {m.annualized_vol:>8.3f} "
            f"{m.sharpe_ratio:>8.3f} {m.max_drawdown:>8.3f} {m.turnover:>10.3f}"
        )

    return "\n".join(lines)


def format_ablation_summary(result: ValidationResult) -> str:
    """Format ablation summary showing marginal Sharpe per component.

    Marginal Sharpe = Sharpe(full model) - Sharpe(model without component).
    Positive = component helps. Negative = component may be hurting.

    Args:
        result: Single-seed validation result.

    Returns:
        Formatted string summary.
    """
    lines = []
    lines.append("Ablation Summary (marginal Sharpe per component):")
    lines.append(f"{'Component':<12} {'Marginal Sharpe':>16} {'Assessment':>12}")
    lines.append("-" * 42)

    for component, marginal in sorted(result.ablation.items()):
        if marginal > 0.05:
            assessment = "helpful"
        elif marginal < -0.05:
            assessment = "may hurt"
        else:
            assessment = "neutral"

        lines.append(f"{component:<12} {marginal:>16.4f} {assessment:>12}")

    return "\n".join(lines)


def format_multi_seed_summary(results: list[ValidationResult]) -> str:
    """Format summary across multiple seeds.

    Args:
        results: List of ValidationResult from different seeds.

    Returns:
        Formatted multi-seed summary.
    """
    lines = []
    n = len(results)
    lines.append(f"Multi-seed validation ({n} seeds)")
    lines.append("=" * 60)

    # Collect Sharpe ratios
    full_sharpes = []
    ew_sharpes = []

    for r in results:
        for v in r.variants:
            if v.name == "full_model":
                full_sharpes.append(v.metrics.sharpe_ratio)
        for b in r.baselines:
            if b.name == "equal_weight":
                ew_sharpes.append(b.metrics.sharpe_ratio)

    if full_sharpes:
        lines.append(
            f"CGF full model Sharpe:  mean={np.mean(full_sharpes):.4f}, "
            f"std={np.std(full_sharpes):.4f}, "
            f"min={np.min(full_sharpes):.4f}, max={np.max(full_sharpes):.4f}"
        )
    if ew_sharpes:
        lines.append(
            f"Equal-weight Sharpe:    mean={np.mean(ew_sharpes):.4f}, "
            f"std={np.std(ew_sharpes):.4f}, "
            f"min={np.min(ew_sharpes):.4f}, max={np.max(ew_sharpes):.4f}"
        )

    if full_sharpes and ew_sharpes:
        diffs = [f - e for f, e in zip(full_sharpes, ew_sharpes, strict=True)]
        lines.append(
            f"CGF - EW diff:          mean={np.mean(diffs):.4f}, "
            f"std={np.std(diffs):.4f}"
        )
        n_worse = sum(1 for d in diffs if d < -0.5)
        lines.append(
            f"Seeds where CGF > -0.5 worse: {n - n_worse}/{n}"
        )

    # Ablation across seeds
    lines.append("")
    lines.append("Mean ablation (marginal Sharpe) across seeds:")
    all_components: dict[str, list[float]] = {}
    for r in results:
        for comp, val in r.ablation.items():
            all_components.setdefault(comp, []).append(val)

    for comp in sorted(all_components):
        vals = all_components[comp]
        lines.append(f"  {comp:<8} mean={np.mean(vals):>8.4f}  std={np.std(vals):>8.4f}")

    return "\n".join(lines)


def format_signal_summary(results: list[ValidationResult]) -> str:
    """Format multi-seed summary for signal-embedded validation.

    Shows per-component marginal Sharpe with std and win rate across seeds.

    Args:
        results: List of ValidationResult from signal-embedded runs.

    Returns:
        Formatted signal validation summary.
    """
    lines = []
    n = len(results)
    lines.append(f"Signal-embedded validation ({n} seeds)")
    lines.append("=" * 65)

    # Collect Sharpe ratios
    full_sharpes = []
    ew_sharpes = []

    for r in results:
        for v in r.variants:
            if v.name == "full_model":
                full_sharpes.append(v.metrics.sharpe_ratio)
        for b in r.baselines:
            if b.name == "equal_weight":
                ew_sharpes.append(b.metrics.sharpe_ratio)

    if full_sharpes:
        lines.append(
            f"CGF full model Sharpe:  mean={np.mean(full_sharpes):.4f}, "
            f"std={np.std(full_sharpes):.4f}"
        )
    if ew_sharpes:
        lines.append(
            f"Equal-weight Sharpe:    mean={np.mean(ew_sharpes):.4f}, "
            f"std={np.std(ew_sharpes):.4f}"
        )

    if full_sharpes and ew_sharpes:
        diffs = [f - e for f, e in zip(full_sharpes, ew_sharpes, strict=True)]
        mean_diff = np.mean(diffs)
        lines.append(
            f"CGF - EW diff:          mean={mean_diff:.4f}, "
            f"std={np.std(diffs):.4f}"
        )
        n_better = sum(1 for d in diffs if d > 0)
        lines.append(f"Seeds where CGF beats EW: {n_better}/{n}")

    # Per-component ablation with win rate
    lines.append("")
    lines.append(
        f"{'Component':<10} {'Mean Marg Sharpe':>16} {'Std':>8} "
        f"{'Win Rate':>10} {'Assessment':>12}"
    )
    lines.append("-" * 58)

    all_components: dict[str, list[float]] = {}
    for r in results:
        for comp, val in r.ablation.items():
            all_components.setdefault(comp, []).append(val)

    for comp in sorted(all_components):
        vals = all_components[comp]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        win_rate = sum(1 for v in vals if v > 0) / len(vals) if vals else 0.0

        if mean_val > 0.05 and win_rate > 0.5:
            assessment = "helpful"
        elif mean_val < -0.05:
            assessment = "may hurt"
        else:
            assessment = "neutral"

        lines.append(
            f"{comp:<10} {mean_val:>16.4f} {std_val:>8.4f} "
            f"{win_rate:>9.0%} {assessment:>12}"
        )

    return "\n".join(lines)
