"""Validation report formatting.

Formats comparison tables and ablation summaries for console output.
Supports synthetic, signal-embedded, and real data validation results.
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


def format_real_data_summary(
    result: ValidationResult,
    tickers: list[str] | None = None,
) -> str:
    """Format summary for real data validation.

    Shows universe info, CGF vs baselines, ablation, and per-asset
    conviction statistics.

    Args:
        result: ValidationResult from real data run.
        tickers: Ticker list for display.

    Returns:
        Formatted real data validation summary.
    """
    lines = []
    lines.append("Real Data Validation")
    lines.append("=" * 65)

    if tickers:
        lines.append(f"Universe: {', '.join(tickers)}")
        lines.append("")

    # CGF vs baselines
    full_sharpe = None
    full_ret = None
    full_vol = None
    full_dd = None
    for v in result.variants:
        if v.name == "full_model":
            full_sharpe = v.metrics.sharpe_ratio
            full_ret = v.metrics.annualized_return
            full_vol = v.metrics.annualized_vol
            full_dd = v.metrics.max_drawdown

    ew_sharpe = None
    for b in result.baselines:
        if b.name == "equal_weight":
            ew_sharpe = b.metrics.sharpe_ratio

    if full_sharpe is not None:
        lines.append(
            f"CGF full model:  Sharpe={full_sharpe:.4f}, "
            f"Return={full_ret:.3f}, Vol={full_vol:.3f}, MaxDD={full_dd:.3f}"
        )
    if ew_sharpe is not None:
        lines.append(f"Equal-weight:    Sharpe={ew_sharpe:.4f}")
    if full_sharpe is not None and ew_sharpe is not None:
        diff = full_sharpe - ew_sharpe
        lines.append(f"CGF - EW diff:   {diff:+.4f}")

    # Ablation
    if result.ablation:
        lines.append("")
        lines.append(f"{'Component':<10} {'Marginal Sharpe':>16} {'Assessment':>12}")
        lines.append("-" * 40)
        for comp in sorted(result.ablation):
            marginal = result.ablation[comp]
            if marginal > 0.05:
                assessment = "helpful"
            elif marginal < -0.05:
                assessment = "may hurt"
            else:
                assessment = "neutral"
            lines.append(f"{comp:<10} {marginal:>16.4f} {assessment:>12}")

    # Per-asset conviction stats
    full_convictions = None
    for v in result.variants:
        if v.name == "full_model" and v.convictions:
            full_convictions = v.convictions
            break

    if full_convictions:
        lines.append("")
        lines.append("Per-asset conviction stats (full model):")
        lines.append(
            f"{'Asset':<10} {'Mean C':>8} {'Std C':>8} {'Min C':>8} {'Max C':>8}"
        )
        lines.append("-" * 44)
        for asset_name in sorted(full_convictions):
            c = full_convictions[asset_name]
            if len(c) > 0:
                lines.append(
                    f"{asset_name:<10} {np.mean(c):>8.3f} {np.std(c):>8.3f} "
                    f"{np.min(c):>8.3f} {np.max(c):>8.3f}"
                )

    return "\n".join(lines)
