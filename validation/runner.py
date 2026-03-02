"""Validation runner — drives CGF over synthetic series and computes ablation metrics.

Variants:
- full_model: All 4 components active (FE + FVS + RRS + ADS)
- fe_only: Only forecast error (w2=w3=w4=0)
- no_fe: Zero forecast error weight
- no_fvs: Zero fundamental violation weight
- no_rrs: Zero risk regime shift weight
- no_ads: Zero adversarial debate shift weight
- full_model_adaptive: Full model + adaptive loss weights
- full_model_continuous_regime: Full model + continuous regime detection
- full_model_all_features: Full model + all new features

Signal-embedded mode:
- run_cgf_over_signal_series: Populates ALL InstrumentData fields from
  SignalEmbeddedAsset (forecasts, FVS events, IV, debates, track record).
- run_signal_validation / run_signal_validation_multi_seed: Drive signal-aware
  validation with ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from components.regime_detector import RegimeDetector, RegimeDetectorConfig
from config.defaults import ConvictionParams
from engine.adaptive import AdaptiveWeightTracker
from engine.models import ConvictionState, InstrumentData
from engine.updater import run_single_update
from validation.baselines import (
    PortfolioMetrics,
    buy_and_hold_strategy,
    compute_metrics,
    equal_weight_strategy,
    momentum_strategy,
)
from validation.synthetic_data import SyntheticAsset


@dataclass
class VariantResult:
    """Result for one CGF variant or baseline."""

    name: str
    metrics: PortfolioMetrics
    convictions: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Full validation result across all variants and seeds."""

    variants: list[VariantResult]
    baselines: list[VariantResult]
    ablation: dict[str, float] = field(default_factory=dict)


def _make_params_variant(
    variant: str,
    base_params: ConvictionParams | None = None,
) -> ConvictionParams:
    """Create ConvictionParams for a specific ablation variant.

    Args:
        variant: Variant name.
        base_params: Optional base params to derive from. If None, uses defaults.
    """
    base = base_params or ConvictionParams()
    if variant == "full_model":
        return base
    if variant == "fe_only":
        return base.model_copy(update={
            "w1": 1.0, "w2": 0.0, "w3": 0.0, "w4": 0.0,
            "lambda1": 1.0, "lambda2": 0.0, "lambda3": 0.0, "lambda4": 0.0,
        })
    if variant == "no_fe":
        return base.model_copy(update={"w1": 0.0, "lambda1": 0.0})
    if variant == "no_fvs":
        return base.model_copy(update={"w2": 0.0, "lambda2": 0.0})
    if variant == "no_rrs":
        return base.model_copy(update={"w3": 0.0, "lambda3": 0.0})
    if variant == "no_ads":
        return base.model_copy(update={"w4": 0.0, "lambda4": 0.0})
    if variant == "full_model_adaptive":
        return base.model_copy(update={"adaptive_weights": True, "adaptive_lookback": 63})
    if variant == "full_model_continuous_regime":
        return base.model_copy(update={"continuous_regime": True})
    if variant == "full_model_all_features":
        return base.model_copy(update={
            "adaptive_weights": True, "adaptive_lookback": 63,
            "continuous_regime": True,
        })
    return base


# Variants that need special runner handling (regime detector / adaptive tracker)
_FEATURE_VARIANTS = {
    "full_model_adaptive",
    "full_model_continuous_regime",
    "full_model_all_features",
}

ABLATION_VARIANTS = ["full_model", "fe_only", "no_fe", "no_fvs", "no_rrs", "no_ads"]

FEATURE_VARIANTS = [
    "full_model_adaptive",
    "full_model_continuous_regime",
    "full_model_all_features",
]


def run_cgf_over_series(
    assets: list[SyntheticAsset],
    market_returns: np.ndarray,
    params: ConvictionParams | None = None,
    warmup: int = 22,
    regime_detector: RegimeDetector | None = None,
    adaptive_tracker: AdaptiveWeightTracker | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run CGF conviction updates over synthetic return series.

    Uses a rolling window to construct InstrumentData for each day,
    then maps convictions to simple conviction-proportional weights.

    Args:
        assets: List of synthetic assets.
        market_returns: Market return series for vol computation.
        params: ConvictionParams (default if None).
        warmup: Days before starting updates (need history for vol).
        regime_detector: Optional RegimeDetector for continuous regime-aware RRS.
        adaptive_tracker: Optional AdaptiveWeightTracker for adaptive loss weights.

    Returns:
        Tuple of (portfolio daily returns array, dict of conviction arrays per asset).
    """
    p = params or ConvictionParams()
    n_days = min(len(a.returns) for a in assets)
    n_assets = len(assets)

    if n_days <= warmup or n_assets == 0:
        return np.zeros(max(n_days - warmup, 0)), {}

    # Initialize states
    base_date = date(2020, 1, 1)
    states: dict[str, ConvictionState] = {}
    for asset in assets:
        states[asset.name] = ConvictionState(
            instrument_id=asset.name,
            as_of_date=base_date,
            conviction=0.0,
            conviction_prev=0.0,
        )

    active_days = n_days - warmup
    port_returns = np.zeros(active_days)
    conviction_history: dict[str, list[float]] = {a.name: [] for a in assets}

    for t in range(warmup, n_days):
        day_idx = t - warmup
        as_of = base_date + timedelta(days=t)

        # Compute vol estimates from trailing window
        convictions = {}
        for asset in assets:
            trail = asset.returns[max(0, t - 62) : t]

            sigma_current = (
                float(np.std(trail, ddof=1)) * np.sqrt(252) if len(trail) > 1 else 0.2
            )
            sigma_prev = sigma_current * 0.98  # Slight lag proxy

            # Simple expected return: trailing mean annualized
            exp_ret = float(np.mean(trail)) * 252 if len(trail) > 0 else 0.0
            sigma_exp = max(sigma_current, 0.01)

            data = InstrumentData(
                instrument_id=asset.name,
                as_of_date=as_of,
                realized_return=float(asset.returns[t]),
                expected_return=exp_ret,
                sigma_expected=sigma_exp,
                sigma_idio_current=max(sigma_current, 0.01),
                sigma_idio_prev=max(sigma_prev, 0.01),
            )

            new_state = run_single_update(
                states[asset.name], data, p,
                regime_detector=regime_detector,
                adaptive_tracker=adaptive_tracker,
            )
            states[asset.name] = new_state
            convictions[asset.name] = new_state.conviction
            conviction_history[asset.name].append(new_state.conviction)

        # Map convictions to weights (conviction-proportional)
        c_vals = np.array([convictions[a.name] for a in assets])
        abs_sum = np.sum(np.abs(c_vals))
        weights = c_vals / abs_sum if abs_sum > 1e-10 else np.ones(n_assets) / n_assets

        # Portfolio return
        day_returns = np.array([a.returns[t] for a in assets])
        port_returns[day_idx] = float(np.dot(weights, day_returns))

    conv_arrays = {k: np.array(v) for k, v in conviction_history.items()}
    return port_returns, conv_arrays


def run_validation(
    assets: list[SyntheticAsset],
    market_returns: np.ndarray,
    variants: list[str] | None = None,
) -> ValidationResult:
    """Run full validation: CGF variants + baselines.

    Args:
        assets: Synthetic asset universe.
        market_returns: Market return series.
        variants: List of variant names to run (defaults to all ablation variants).

    Returns:
        ValidationResult with all variant and baseline results.
    """
    variants = variants or ABLATION_VARIANTS

    # Stack asset returns for baseline strategies
    n_days = min(len(a.returns) for a in assets)
    asset_matrix = np.column_stack([a.returns[:n_days] for a in assets])

    # Run baselines
    baselines = []

    ew_ret, ew_turn = equal_weight_strategy(asset_matrix)
    baselines.append(VariantResult(
        name="equal_weight",
        metrics=compute_metrics(ew_ret, ew_turn),
    ))

    bh_ret, bh_turn = buy_and_hold_strategy(asset_matrix)
    baselines.append(VariantResult(
        name="buy_and_hold",
        metrics=compute_metrics(bh_ret, bh_turn),
    ))

    mom_ret, mom_turn = momentum_strategy(asset_matrix)
    baselines.append(VariantResult(
        name="momentum",
        metrics=compute_metrics(mom_ret, mom_turn),
    ))

    # Run CGF variants
    variant_results = []
    for variant_name in variants:
        params = _make_params_variant(variant_name)

        # Set up feature objects if needed
        regime_det = None
        adapt_tracker = None
        if variant_name in _FEATURE_VARIANTS:
            if params.continuous_regime:
                regime_det = RegimeDetector(RegimeDetectorConfig(
                    vol_threshold=params.regime_vol_threshold,
                    transition_penalty=params.regime_transition_penalty,
                ))
            if params.adaptive_weights:
                adapt_tracker = AdaptiveWeightTracker(
                    lookback=params.adaptive_lookback,
                    decay=params.adaptive_decay,
                )

        port_ret, convictions = run_cgf_over_series(
            assets, market_returns, params,
            regime_detector=regime_det,
            adaptive_tracker=adapt_tracker,
        )
        variant_results.append(VariantResult(
            name=variant_name,
            metrics=compute_metrics(port_ret),
            convictions=convictions,
        ))

    # Compute ablation: marginal Sharpe = Sharpe(full) - Sharpe(no_X)
    ablation = {}
    full_sharpe = None
    for vr in variant_results:
        if vr.name == "full_model":
            full_sharpe = vr.metrics.sharpe_ratio
            break

    if full_sharpe is not None:
        for vr in variant_results:
            if vr.name.startswith("no_"):
                component = vr.name[3:]  # e.g. "fe", "fvs", "rrs", "ads"
                ablation[component] = full_sharpe - vr.metrics.sharpe_ratio

    return ValidationResult(
        variants=variant_results,
        baselines=baselines,
        ablation=ablation,
    )


def run_validation_multi_seed(
    n_seeds: int = 5,
    n_days: int = 504,
    base_seed: int = 42,
) -> list[ValidationResult]:
    """Run validation across multiple seeds for robustness.

    Args:
        n_seeds: Number of random seeds.
        n_days: Trading days per seed.
        base_seed: Starting seed.

    Returns:
        List of ValidationResult, one per seed.
    """
    from validation.synthetic_data import generate_universe

    results = []
    for i in range(n_seeds):
        seed = base_seed + i * 1000
        assets, market = generate_universe(n_days=n_days, seed=seed)
        result = run_validation(assets, market)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Signal-embedded validation
# ---------------------------------------------------------------------------

SIGNAL_VARIANTS = ABLATION_VARIANTS  # Same ablation structure


def run_cgf_over_signal_series(
    assets: list,
    market_returns: np.ndarray,
    params: ConvictionParams | None = None,
    warmup: int = 22,
    regime_detector: RegimeDetector | None = None,
    adaptive_tracker: AdaptiveWeightTracker | None = None,
    initial_conviction: float = 2.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run CGF conviction updates over signal-embedded return series.

    Unlike run_cgf_over_series, this populates ALL InstrumentData fields
    from SignalEmbeddedAsset: forecasts, FVS events, implied vol, debate
    shifts, and track record score. Uses daily sigma (not annualized)
    for sigma_expected to match daily returns and daily forecasts.

    Args:
        assets: List of SignalEmbeddedAsset.
        market_returns: Market return series.
        params: ConvictionParams (default if None).
        warmup: Days before starting updates.
        regime_detector: Optional RegimeDetector.
        adaptive_tracker: Optional AdaptiveWeightTracker.
        initial_conviction: Starting conviction for all assets.

    Returns:
        Tuple of (portfolio daily returns array, dict of conviction arrays).
    """
    p = params or ConvictionParams()
    n_days = min(len(a.returns) for a in assets)
    n_assets = len(assets)

    if n_days <= warmup or n_assets == 0:
        return np.zeros(max(n_days - warmup, 0)), {}

    # Pre-index FVS and debate schedules by day for fast lookup
    fvs_by_day: dict[str, dict[int, list[dict]]] = {}
    debate_by_day: dict[str, dict[int, dict]] = {}
    for asset in assets:
        fvs_lookup: dict[int, list[dict]] = {}
        for evt in asset.fvs_schedule:
            day = evt["day"]
            fvs_lookup.setdefault(day, []).append(evt)
        fvs_by_day[asset.name] = fvs_lookup

        debate_lookup: dict[int, dict] = {}
        for deb in asset.debate_schedule:
            debate_lookup[deb["day"]] = deb
        debate_by_day[asset.name] = debate_lookup

    # Initialize states with non-zero conviction (existing positions)
    base_date = date(2020, 1, 1)
    states: dict[str, ConvictionState] = {}
    for asset in assets:
        states[asset.name] = ConvictionState(
            instrument_id=asset.name,
            as_of_date=base_date,
            conviction=initial_conviction,
            conviction_prev=initial_conviction,
        )

    active_days = n_days - warmup
    port_returns = np.zeros(active_days)
    conviction_history: dict[str, list[float]] = {a.name: [] for a in assets}

    for t in range(warmup, n_days):
        day_idx = t - warmup
        as_of = base_date + timedelta(days=t)

        convictions = {}
        for asset in assets:
            trail = asset.returns[max(0, t - 62) : t]

            # Daily sigma (NOT annualized) — matches daily returns/forecasts
            daily_sigma = (
                float(np.std(trail, ddof=1)) if len(trail) > 1 else 0.01
            )
            daily_sigma = max(daily_sigma, 1e-6)

            # Previous period daily sigma (slight lag)
            prev_trail = asset.returns[max(0, t - 63) : max(0, t - 1)]
            daily_sigma_prev = (
                float(np.std(prev_trail, ddof=1)) if len(prev_trail) > 1 else daily_sigma
            )
            daily_sigma_prev = max(daily_sigma_prev, 1e-6)

            # Expected return from embedded forecast (daily)
            exp_ret = float(asset.forecasts[t]) if t < len(asset.forecasts) else 0.0

            # FVS events for this day
            fvs_events = []
            day_events = fvs_by_day[asset.name].get(t, [])
            for evt in day_events:
                fvs_events.append({
                    "event_type": evt["event_type"],
                    "severity_override": evt["severity"],
                    "event_date": as_of.isoformat(),
                    "description": f"Synthetic {evt['event_type']}",
                })

            # Implied vol and historical vol
            iv = (
                float(asset.implied_vols[t])
                if t < len(asset.implied_vols)
                else None
            )
            # Historical vol = trailing realized daily vol
            hv = daily_sigma if daily_sigma > 1e-6 else None

            # Debate data for this day
            debate = debate_by_day[asset.name].get(t)
            p_pre = debate["p_pre"] if debate else []
            p_post = debate["p_post"] if debate else []

            data = InstrumentData(
                instrument_id=asset.name,
                as_of_date=as_of,
                realized_return=float(asset.returns[t]),
                expected_return=exp_ret,
                sigma_expected=daily_sigma,
                sigma_idio_current=daily_sigma,
                sigma_idio_prev=daily_sigma_prev,
                implied_vol=iv,
                historical_vol=hv,
                fvs_events=fvs_events,
                p_pre=p_pre,
                p_post=p_post,
                track_record_score=asset.track_record_score,
            )

            new_state = run_single_update(
                states[asset.name], data, p,
                regime_detector=regime_detector,
                adaptive_tracker=adaptive_tracker,
            )
            states[asset.name] = new_state
            convictions[asset.name] = new_state.conviction
            conviction_history[asset.name].append(new_state.conviction)

        # Map convictions to weights (conviction-proportional)
        c_vals = np.array([convictions[a.name] for a in assets])
        abs_sum = np.sum(np.abs(c_vals))
        weights = c_vals / abs_sum if abs_sum > 1e-10 else np.ones(n_assets) / n_assets

        # Portfolio return
        day_returns = np.array([a.returns[t] for a in assets])
        port_returns[day_idx] = float(np.dot(weights, day_returns))

    conv_arrays = {k: np.array(v) for k, v in conviction_history.items()}
    return port_returns, conv_arrays


def run_signal_validation(
    assets: list,
    market_returns: np.ndarray,
    variants: list[str] | None = None,
) -> ValidationResult:
    """Run validation over signal-embedded data: CGF variants + baselines.

    Uses a base ConvictionParams with kappa tuned for daily frequency.
    The default kappa=0.1 is designed for periodic (monthly/quarterly) updates;
    at daily frequency we need higher kappa to make conviction responsive
    enough that signal differences across ablation variants are visible.

    Args:
        assets: Signal-embedded asset universe.
        market_returns: Market return series.
        variants: Variant names to run (defaults to SIGNAL_VARIANTS).

    Returns:
        ValidationResult with all variant and baseline results.
    """
    variants = variants or SIGNAL_VARIANTS

    # Base params tuned for daily-frequency signal data
    signal_base = ConvictionParams(kappa=0.5)

    # Stack asset returns for baseline strategies
    n_days = min(len(a.returns) for a in assets)
    asset_matrix = np.column_stack([a.returns[:n_days] for a in assets])

    # Run baselines
    baselines = []

    ew_ret, ew_turn = equal_weight_strategy(asset_matrix)
    baselines.append(VariantResult(
        name="equal_weight",
        metrics=compute_metrics(ew_ret, ew_turn),
    ))

    bh_ret, bh_turn = buy_and_hold_strategy(asset_matrix)
    baselines.append(VariantResult(
        name="buy_and_hold",
        metrics=compute_metrics(bh_ret, bh_turn),
    ))

    mom_ret, mom_turn = momentum_strategy(asset_matrix)
    baselines.append(VariantResult(
        name="momentum",
        metrics=compute_metrics(mom_ret, mom_turn),
    ))

    # Run CGF variants over signal-embedded data
    variant_results = []
    for variant_name in variants:
        params = _make_params_variant(variant_name, base_params=signal_base)

        regime_det = None
        adapt_tracker = None
        if variant_name in _FEATURE_VARIANTS:
            if params.continuous_regime:
                regime_det = RegimeDetector(RegimeDetectorConfig(
                    vol_threshold=params.regime_vol_threshold,
                    transition_penalty=params.regime_transition_penalty,
                ))
            if params.adaptive_weights:
                adapt_tracker = AdaptiveWeightTracker(
                    lookback=params.adaptive_lookback,
                    decay=params.adaptive_decay,
                )

        port_ret, convictions = run_cgf_over_signal_series(
            assets, market_returns, params,
            regime_detector=regime_det,
            adaptive_tracker=adapt_tracker,
        )
        variant_results.append(VariantResult(
            name=variant_name,
            metrics=compute_metrics(port_ret),
            convictions=convictions,
        ))

    # Compute ablation: marginal Sharpe = Sharpe(full) - Sharpe(no_X)
    ablation = {}
    full_sharpe = None
    for vr in variant_results:
        if vr.name == "full_model":
            full_sharpe = vr.metrics.sharpe_ratio
            break

    if full_sharpe is not None:
        for vr in variant_results:
            if vr.name.startswith("no_"):
                component = vr.name[3:]
                ablation[component] = full_sharpe - vr.metrics.sharpe_ratio

    return ValidationResult(
        variants=variant_results,
        baselines=baselines,
        ablation=ablation,
    )


def run_signal_validation_multi_seed(
    n_seeds: int = 5,
    n_days: int = 504,
    base_seed: int = 42,
) -> list[ValidationResult]:
    """Run signal-embedded validation across multiple seeds.

    Args:
        n_seeds: Number of random seeds.
        n_days: Trading days per seed.
        base_seed: Starting seed.

    Returns:
        List of ValidationResult, one per seed.
    """
    from validation.signal_data import generate_signal_universe

    results = []
    for i in range(n_seeds):
        seed = base_seed + i * 1000
        assets, market = generate_signal_universe(n_days=n_days, seed=seed)
        result = run_signal_validation(assets, market)
        results.append(result)
    return results
