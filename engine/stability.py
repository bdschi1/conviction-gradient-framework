"""Stability checks, clipping, and oscillation detection.

Conviction dynamics remain bounded if 0 < alpha * lambda_max < 2.
Oscillation detection flags conviction sign flips within a window.
"""

from __future__ import annotations


def apply_clipping(conviction: float, c_max: float) -> float:
    """Clamp conviction to [-c_max, c_max].

    Args:
        conviction: Raw conviction value.
        c_max: Maximum absolute conviction.

    Returns:
        Clipped conviction.
    """
    return max(-c_max, min(conviction, c_max))


def check_stability(alpha: float, lambda_max: float) -> bool:
    """Check if conviction update satisfies the stability condition.

    The update is stable when 0 < alpha * lambda_max < 2.

    Args:
        alpha: Current learning rate.
        lambda_max: Largest eigenvalue of the loss Hessian (or max lambda_i).

    Returns:
        True if stable, False otherwise.
    """
    product = alpha * lambda_max
    return 0 < product < 2


def detect_oscillation(
    conviction_history: list[float],
    window: int = 5,
) -> bool:
    """Detect if conviction has been oscillating (sign flips).

    Args:
        conviction_history: Recent conviction values, most recent last.
        window: Number of recent values to check.

    Returns:
        True if a sign flip occurred within the window.
    """
    if len(conviction_history) < 2:
        return False

    recent = conviction_history[-window:]
    return any(recent[i] * recent[i - 1] < 0 for i in range(1, len(recent)))


def count_sign_flips(conviction_history: list[float], window: int = 5) -> int:
    """Count number of sign flips in recent conviction history.

    Args:
        conviction_history: Recent conviction values, most recent last.
        window: Number of recent values to check.

    Returns:
        Number of sign flips in the window.
    """
    if len(conviction_history) < 2:
        return 0

    recent = conviction_history[-window:]
    flips = 0
    for i in range(1, len(recent)):
        if recent[i] * recent[i - 1] < 0:
            flips += 1

    return flips
