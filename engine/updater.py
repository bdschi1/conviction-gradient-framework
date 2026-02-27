"""Core conviction update rule.

C_{t+1} = C_t - alpha_t * nabla_L_t + beta * (C_t - C_{t-1})

The update combines gradient descent (move conviction away from thesis error)
with momentum (smooth out noisy updates and prevent oscillation).
Clipping prevents overconfidence or panic.
"""

from __future__ import annotations

import logging

from components.debate_shift import compute_ads
from components.forecast_error import compute_fe
from components.fundamental_violation import FVSEvent, FVSTaxonomy, compute_fvs
from components.risk_regime import compute_rrs
from config.defaults import ConvictionParams
from engine.gradient import compute_gradient_result, compute_learning_rate
from engine.loss import compute_loss
from engine.models import ConvictionState, InstrumentData
from engine.stability import apply_clipping

logger = logging.getLogger(__name__)


def update_conviction(
    current: ConvictionState,
    gradient_value: float,
    alpha_t: float,
    beta: float,
    c_max: float,
) -> float:
    """Apply the core conviction update rule.

    Args:
        current: Current conviction state.
        gradient_value: Computed gradient nabla L_t.
        alpha_t: Adaptive learning rate.
        beta: Momentum stabilizer.
        c_max: Conviction clamp magnitude.

    Returns:
        Updated conviction value, clipped to [-c_max, c_max].
    """
    momentum = beta * (current.conviction - current.conviction_prev)
    new_c = current.conviction - alpha_t * gradient_value + momentum
    return apply_clipping(new_c, c_max)


def run_single_update(
    current: ConvictionState,
    data: InstrumentData,
    params: ConvictionParams | None = None,
    taxonomy: FVSTaxonomy | None = None,
) -> ConvictionState:
    """Run a full conviction update for one instrument.

    Computes all loss components, gradient, learning rate, and produces
    the next conviction state.

    Args:
        current: Current conviction state.
        data: New data for this instrument.
        params: Hyperparameters.
        taxonomy: FVS event taxonomy.

    Returns:
        New ConvictionState with updated conviction.
    """
    p = params or ConvictionParams()

    # Compute loss components
    fe = compute_fe(data.realized_return, data.expected_return, data.sigma_expected)
    fvs_events = [FVSEvent(**e) if isinstance(e, dict) else e for e in data.fvs_events]
    fvs = compute_fvs(fvs_events, taxonomy)
    rrs = compute_rrs(
        data.sigma_idio_current,
        data.sigma_idio_prev,
        data.implied_vol,
        data.historical_vol,
    )
    ads = compute_ads(data.p_pre, data.p_post)

    # Compute loss and gradient
    loss = compute_loss(fe, fvs, rrs, ads, p)

    alpha_t = compute_learning_rate(
        kappa=p.kappa,
        info_half_life=data.info_half_life,
        sigma_idio=data.sigma_idio_current,
        sigma_expected=data.sigma_expected,
        track_record_score=data.track_record_score,
        alpha_min=p.alpha_min,
        alpha_max=p.alpha_max,
    )

    gradient = compute_gradient_result(fe, fvs, rrs, ads, alpha_t, p)

    # Update conviction
    new_c = update_conviction(current, gradient.gradient_value, alpha_t, p.beta, p.C_max)

    return ConvictionState(
        instrument_id=data.instrument_id,
        as_of_date=data.as_of_date,
        conviction=new_c,
        conviction_prev=current.conviction,
        expected_return=data.expected_return,
        idiosyncratic_vol=data.sigma_idio_current,
        alpha_t=alpha_t,
        loss_components=loss,
        gradient=gradient,
    )


def run_batch_update(
    states: dict[str, ConvictionState],
    data_batch: list[InstrumentData],
    params: ConvictionParams | None = None,
    taxonomy: FVSTaxonomy | None = None,
) -> dict[str, ConvictionState]:
    """Run conviction updates for a batch of instruments.

    Args:
        states: Current conviction states keyed by instrument_id.
        data_batch: New data for each instrument.
        params: Shared hyperparameters.
        taxonomy: FVS event taxonomy.

    Returns:
        Updated states keyed by instrument_id.
    """
    p = params or ConvictionParams()
    results = {}

    for data in data_batch:
        current = states.get(data.instrument_id)
        if current is None:
            # Initialize new position with zero conviction
            current = ConvictionState(
                instrument_id=data.instrument_id,
                as_of_date=data.as_of_date,
                conviction=0.0,
                conviction_prev=0.0,
            )

        results[data.instrument_id] = run_single_update(current, data, p, taxonomy)

    return results
