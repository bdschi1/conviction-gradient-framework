"""CGF Streamlit UI — Conviction Gradient Framework interactive dashboard.

Scenario inputs on the main page, engine parameters in the sidebar,
tabbed results with narrative analysis and a printable portfolio report.
Market data auto-fetched via yfinance; only ticker is required from the user.
Consensus price targets are auto-populated for expected return computation.
"""

from __future__ import annotations

import html
import io
import logging
from datetime import date

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from bridges.data_bridge import (
    VolMethod,
    compute_historical_vol,
    compute_idio_vol,
)
from components.fundamental_violation import FVSTaxonomy
from config.defaults import ConvictionParams
from engine.models import ConvictionState, InstrumentData
from engine.updater import run_batch_update
from sizing.constraints import (
    ConstrainedResult,
    PortfolioConstraints,
    apply_constraints,
)
from sizing.failure_modes import (
    FailureModeAction,
    check_oscillation_guard,
    check_structural_reset,
)
from sizing.mapper import SizingMethod, map_convictions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CGF — Conviction Gradient Framework",
    page_icon="📐",
    layout="wide",
)

TAXONOMY = FVSTaxonomy()
PLOTLY_TEMPLATE = "plotly_dark"
GREEN = "#2ecc71"
RED = "#e74c3c"
GRAY = "#636e72"

VOL_METHOD_LABELS = {
    "CAPM (default)": VolMethod.CAPM,
    "Fama-French 3-Factor": VolMethod.FF3,
    "EWMA (RiskMetrics)": VolMethod.EWMA,
    "GARCH(1,1)": VolMethod.GARCH,
}

SIZING_METHOD_LABELS = {
    "Basic": SizingMethod.BASIC,
    "Vol-Adjusted": SizingMethod.VOL_ADJUSTED,
    "Kelly Fraction": SizingMethod.KELLY,
    "Risk Parity": SizingMethod.RISK_PARITY,
    "Tiered": SizingMethod.TIERED,
}


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
def _init_state():
    if "instruments" not in st.session_state:
        st.session_state.instruments = []
    if "run_complete" not in st.session_state:
        st.session_state.run_complete = False
    if "results" not in st.session_state:
        st.session_state.results = {}


_init_state()


# ---------------------------------------------------------------------------
# Auto-fetch market data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Fetching market data...")
def _fetch_instrument_data(
    ticker: str,
    vol_method: str = "CAPM (default)",
) -> dict | None:
    """Fetch market data for a ticker via yfinance.

    Returns a dict with realized_return, sigma_expected,
    sigma_idio_current, sigma_idio_prev, historical_vol, sector,
    target_price, current_price, n_analysts, or None on failure.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed")
        return None

    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        sector = info.get("sector", "Unknown")

        # Consensus price target
        target_price = info.get("targetMeanPrice")
        current_price = info.get("currentPrice")
        n_analysts = info.get("numberOfAnalystOpinions", 0)

        hist = tk.history(period="1y")
        if hist.empty or len(hist) < 30:
            return None

        prices = hist["Close"].tolist()
        returns = [
            (prices[i] / prices[i - 1]) - 1
            for i in range(1, len(prices))
        ]

        # Fallback current_price from history if info didn't have it
        if current_price is None:
            current_price = prices[-1]

        # SPY for CAPM/EWMA/GARCH regression
        spy_hist = yf.Ticker("SPY").history(period="1y")
        spy_prices = spy_hist["Close"].tolist()
        spy_returns = [
            (spy_prices[i] / spy_prices[i - 1]) - 1
            for i in range(1, len(spy_prices))
        ]

        # Align lengths
        n = min(len(returns), len(spy_returns))
        returns = returns[-n:]
        spy_returns = spy_returns[-n:]

        # Realized return (trailing ~63 trading days)
        n_qtr = min(63, len(prices) - 1)
        realized_return = (prices[-1] / prices[-(n_qtr + 1)]) - 1

        # Sigma expected: annualized vol over ~6 months
        n_half = min(126, len(returns))
        sigma_expected = compute_historical_vol(
            returns[-n_half:], window=n_half
        )

        # Resolve vol method
        vm = VOL_METHOD_LABELS.get(vol_method, VolMethod.CAPM)

        # Sigma idio current (recent 63 days)
        n_idio = min(63, n)
        sigma_idio_current = compute_idio_vol(
            vm, returns[-n_idio:], spy_returns[-n_idio:]
        )

        # Sigma idio prev (prior 63-day window)
        if n >= 126:
            sigma_idio_prev = compute_idio_vol(
                vm, returns[-126:-63], spy_returns[-126:-63]
            )
        else:
            sigma_idio_prev = sigma_idio_current

        # Historical vol (21 day)
        historical_vol = compute_historical_vol(returns, window=21)

        # Guard against zero/nan
        for val in [
            sigma_expected, sigma_idio_current,
            sigma_idio_prev, historical_vol,
        ]:
            if val is None or np.isnan(val) or val <= 0:
                return None

        return {
            "realized_return": round(realized_return, 6),
            "sigma_expected": round(sigma_expected, 6),
            "sigma_idio_current": round(sigma_idio_current, 6),
            "sigma_idio_prev": round(sigma_idio_prev, 6),
            "historical_vol": round(historical_vol, 6),
            "sector": sector,
            "target_price": target_price,
            "current_price": round(current_price, 2) if current_price else None,
            "n_analysts": n_analysts or 0,
        }
    except Exception:
        logger.exception("Failed to fetch data for %s", ticker)
        return None


# ---------------------------------------------------------------------------
# Glossary dialog
# ---------------------------------------------------------------------------
@st.dialog("Glossary", width="large")
def _show_glossary():
    """Render glossary of terms, definitions, and equations."""
    st.markdown("### Core Concepts")
    st.markdown(
        "**Conviction (C)** — A signed score representing how "
        "strongly the model believes in a position. Positive = "
        "long, negative = short. Bounded by C_max."
    )
    st.latex(
        r"C_{t+1} = C_t - \alpha_t \cdot \nabla L_t "
        r"+ \beta \cdot (C_t - C_{t-1})"
    )
    st.caption(
        "Each period, conviction moves in the direction that "
        "reduces loss, with momentum for stability."
    )

    st.markdown(
        "**Learning Rate** (alpha) — Controls how aggressively "
        "conviction updates. Adapts based on information quality "
        "and volatility. Bounded between alpha_min and alpha_max."
    )
    st.latex(
        r"\alpha_t = \text{clamp}\!\left("
        r"\kappa \cdot \frac{h}{\sigma_{idio} "
        r"\cdot \sigma_{exp}} \cdot (1 - \tau),"
        r"\;\alpha_{min},\;\alpha_{max}\right)"
    )

    st.markdown(
        "**Momentum** (beta) — Fraction of the prior conviction "
        "change carried forward. Smooths noisy updates."
    )

    st.divider()
    st.markdown("### Loss Components")
    st.markdown(
        "The total loss measures how wrong the current conviction "
        "appears to be, combining four signals:"
    )
    st.latex(
        r"L_t = w_1 \cdot FE^2 + w_2 \cdot FVS "
        r"+ w_3 \cdot |RRS| + w_4 \cdot |ADS|"
    )

    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            "**FE — Forecast Error**\n\n"
            "How much actual returns missed the forecast, "
            "normalized by expected volatility."
        )
        st.latex(
            r"FE = \frac{R_{actual} - R_{expected}}{\sigma_{expected}}"
        )
        st.markdown(
            "**FVS — Fundamental Violation Score**\n\n"
            "A 0–1 score reflecting thesis-breaking events "
            "(management change, guidance cut, governance breach, "
            "etc.). Uses a severity taxonomy."
        )
        st.latex(
            r"FVS = \min\!\left(1,\;\sum_i s_i\right)"
        )
        st.caption("where s_i is the severity of each event.")
    with cols[1]:
        st.markdown(
            "**RRS — Risk Regime Shift**\n\n"
            "Detects changes in the stock's risk profile. "
            "Compares current vs prior idiosyncratic vol, "
            "and implied vs historical vol spread."
        )
        st.latex(
            r"RRS = \frac{\sigma_{idio}^{curr} - \sigma_{idio}^{prev}}"
            r"{\sigma_{idio}^{prev}}"
            r" + \lambda_{iv}\left(\sigma_{IV} - \sigma_{HV}\right)"
        )
        st.markdown(
            "**ADS — Adversarial Debate Shift**\n\n"
            "Captures how an IC debate moved probability "
            "estimates. A large shift means the debate "
            "changed minds."
        )
        st.latex(
            r"ADS = \overline{p}_{post} - \overline{p}_{pre}"
        )

    st.divider()
    st.markdown("### Volatility Terms")
    st.markdown(
        "**Sigma Expected** (sigma_exp) — Annualized historical "
        "volatility over ~6 months. Represents how much return "
        "variation is *normal* for this stock. Used to normalize "
        "forecast error so that a 5% miss on a 40%-vol biotech "
        "is treated differently from a 5% miss on a 12%-vol "
        "utility."
    )
    st.markdown(
        "**Sigma Idiosyncratic** (sigma_idio) — The portion of "
        "a stock's volatility that is *not* explained by the "
        "market (SPY). Computed via CAPM regression: regress "
        "stock returns on market returns, take std of residuals. "
        "'Current' uses the most recent ~63 trading days; "
        "'Previous' uses the prior ~63-day window. Comparing "
        "them detects whether the stock is becoming riskier "
        "independent of the market."
    )
    st.markdown(
        "**Historical Vol** — Realized annualized volatility "
        "over the trailing 21 trading days (~1 month)."
    )

    st.divider()
    st.markdown("### Idiosyncratic Vol Methods")

    st.markdown(
        "**CAPM** — Single-factor regression vs SPY. "
        "Residual std dev = idiosyncratic vol."
    )
    st.latex(
        r"r_i = \alpha + \beta \, r_m + \varepsilon_i"
        r"\qquad\Rightarrow\qquad"
        r"\sigma_{idio} = \text{std}(\varepsilon) \cdot \sqrt{252}"
    )

    st.markdown(
        "**Fama-French 3-Factor** — Regresses stock returns against "
        "Mkt-RF, SMB (size), and HML (value). Residual captures "
        "vol not explained by market, size, or value."
    )
    st.latex(
        r"r_i - r_f = \alpha + \beta_1 (r_m - r_f)"
        r" + \beta_2 \, SMB + \beta_3 \, HML + \varepsilon_i"
    )

    st.markdown(
        "**EWMA (RiskMetrics)** — Exponential decay on squared "
        "CAPM residuals. More responsive to recent vol changes."
    )
    st.latex(
        r"\hat{\sigma}^2_t = \lambda \, \hat{\sigma}^2_{t-1}"
        r" + (1 - \lambda) \, \varepsilon^2_t"
        r"\qquad (\lambda = 0.94)"
    )

    st.markdown(
        "**GARCH(1,1)** — Fits a GARCH model to CAPM residuals "
        "and forecasts next-period conditional variance. "
        "Forward-looking — predicts tomorrow's vol."
    )
    st.latex(
        r"\sigma^2_t = \omega + \alpha_1 \, \varepsilon^2_{t-1}"
        r" + \beta_1 \, \sigma^2_{t-1}"
    )

    st.divider()
    st.markdown("### Sizing Methods")
    st.markdown(
        "**Gross Exposure** — Sum of absolute position weights. "
        "200% gross = $1 long + $1 short per $1 capital."
    )
    st.markdown(
        "**Net Exposure** — Sum of signed weights. "
        "+30% net = moderately long-biased."
    )

    st.markdown(
        "**Basic** — Weight proportional to conviction magnitude."
    )
    st.latex(
        r"w_i = \frac{C_i}{\sum_j |C_j|}"
    )

    st.markdown(
        "**Vol-Adjusted** — Scales by inverse idio vol so "
        "high-vol names get smaller positions."
    )
    st.latex(
        r"w_i = \frac{C_i \,/\, \sigma_i}"
        r"{\sum_j |C_j \,/\, \sigma_j|}"
    )

    st.markdown(
        "**Kelly Fraction** — Optimal bet sizing from information "
        "theory, capped at half-Kelly (industry convention)."
    )
    st.latex(
        r"w_i = \frac{1}{2} \cdot "
        r"\frac{|C_i| \cdot E[R_i]}{\sigma_i^2}"
        r"\quad\text{(normalized)}"
    )

    st.markdown(
        "**Risk Parity (conviction-scaled)** — Equal risk "
        "contribution, scaled by conviction magnitude."
    )
    st.latex(
        r"w_i \propto \frac{1}{\sigma_i} \cdot "
        r"\frac{|C_i|}{\max_j |C_j|}"
        r"\quad\text{(normalized)}"
    )

    st.markdown(
        "**Tiered** — Discrete conviction bands with fixed "
        "weight tiers. How most discretionary L/S PMs size."
    )
    st.latex(
        r"|C| \geq 3 \rightarrow 4.5\%"
        r"\quad|\quad"
        r"|C| \geq 1.5 \rightarrow 2.5\%"
        r"\quad|\quad"
        r"|C| > 0 \rightarrow 1\%"
    )

    if st.button("Close", type="primary"):
        st.rerun()


# ---------------------------------------------------------------------------
# Sidebar — engine parameters
# ---------------------------------------------------------------------------
def _sidebar():
    st.sidebar.header("Idiosyncratic Vol Method")
    vol_method_label = st.sidebar.selectbox(
        "Vol Method",
        list(VOL_METHOD_LABELS.keys()),
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.header("Sizing Method")
    method_label = st.sidebar.radio(
        "Method",
        list(SIZING_METHOD_LABELS.keys()),
        label_visibility="collapsed",
    )
    sizing_method = SIZING_METHOD_LABELS[method_label]

    with st.sidebar.expander("Engine Parameters"):
        w1 = st.slider(
            "w1 (FE weight)", 0.0, 1.0, 0.30, 0.05,
            help="How much forecast error drives conviction updates. "
            "Higher = larger conviction change when returns miss expectations.",
        )
        w2 = st.slider(
            "w2 (FVS weight)", 0.0, 1.0, 0.25, 0.05,
            help="How much fundamental violations (management changes, "
            "guidance cuts) drive conviction updates.",
        )
        w3 = st.slider(
            "w3 (RRS weight)", 0.0, 1.0, 0.25, 0.05,
            help="How much vol regime shifts drive conviction updates. "
            "Higher = more responsive to changing risk profile.",
        )
        w4 = st.slider(
            "w4 (ADS weight)", 0.0, 1.0, 0.20, 0.05,
            help="How much IC debate outcomes drive conviction updates. "
            "Higher = debates have more impact on positioning.",
        )
        kappa = st.slider(
            "kappa (LR scaling)", 0.01, 1.0, 0.10, 0.01,
            help="Scales the learning rate. Higher = conviction moves "
            "faster per update. Lower = more conservative updates.",
        )
        alpha_min = st.slider(
            "alpha_min", 0.001, 0.1, 0.01, 0.001, format="%.3f",
            help="Floor on learning rate. Prevents updates from "
            "becoming so small that conviction never moves.",
        )
        alpha_max = st.slider(
            "alpha_max", 0.1, 1.0, 0.50, 0.05,
            help="Ceiling on learning rate. Prevents any single "
            "update from swinging conviction too aggressively.",
        )
        beta = st.slider(
            "beta (momentum)", 0.0, 1.0, 0.10, 0.05,
            help="Fraction of prior conviction change carried forward. "
            "Higher = smoother updates, less noise sensitivity.",
        )
        c_max = st.slider(
            "C_max (clamp)", 1.0, 10.0, 5.0, 0.5,
            help="Maximum absolute conviction. Conviction is clipped "
            "to [-C_max, C_max] to prevent overconfidence.",
        )

    with st.sidebar.expander("Constraints"):
        max_gross = st.number_input(
            "Max gross exposure", 0.5, 5.0, 2.0, 0.1,
            help="Sum of absolute weights. 2.0 = 200% gross "
            "(e.g. $1 long + $1 short per $1 capital).",
        )
        max_net = st.number_input(
            "Max net exposure", 0.0, 2.0, 0.5, 0.1,
            help="Max directional bias. 0.5 = portfolio can be at "
            "most 50% net long or net short.",
        )
        max_pos = st.number_input(
            "Max position %", 0.01, 1.0, 0.05, 0.01, format="%.2f",
            help="Largest any single position can be. "
            "0.05 = 5% max per name.",
        )
        max_sect = st.number_input(
            "Max sector net", 0.0, 1.0, 0.25, 0.05, format="%.2f",
            help="Max net exposure in any one sector. "
            "0.25 = no sector more than 25% net.",
        )

    with st.sidebar.expander("Failure Modes"):
        fvs_thresh = st.slider(
            "FVS reset threshold", 0.0, 1.0, 0.70, 0.05,
            help="If FVS exceeds this, conviction resets to zero. "
            "Catches thesis-breaking events (e.g. fraud, CEO departure).",
        )
        vol_thresh = st.slider(
            "Vol double threshold", 1.1, 5.0, 2.0, 0.1,
            help="If current idio vol exceeds prior by this multiple, "
            "conviction resets. 2.0 = vol doubled triggers reset.",
        )

    params = ConvictionParams(
        w1=w1, w2=w2, w3=w3, w4=w4,
        kappa=kappa, alpha_min=alpha_min, alpha_max=alpha_max,
        beta=beta, C_max=c_max,
        fvs_reset_threshold=fvs_thresh,
        vol_double_threshold=vol_thresh,
    )
    constraints = PortfolioConstraints(
        max_gross_exposure=max_gross,
        max_net_exposure=max_net,
        max_position_pct=max_pos,
        max_sector_net=max_sect,
    )
    return sizing_method, vol_method_label, params, constraints


# ---------------------------------------------------------------------------
# Action signals
# ---------------------------------------------------------------------------
def _derive_action(
    state: ConvictionState,
    failure_actions: list[FailureModeAction],
) -> tuple[str, str]:
    """Derive an action signal and color from conviction state.

    Returns (action_label, color) where color is for display.
    """
    c = state.conviction
    c_prev = state.conviction_prev
    mag = abs(c)

    # Check for structural reset / thesis damage
    for fa in failure_actions:
        if fa.action_type == "reset_conviction":
            return "EXIT — thesis break", RED

    # Check for oscillation
    for fa in failure_actions:
        if fa.action_type == "halve_alpha":
            return "HOLD — noisy signal", GRAY

    lc = state.loss_components
    if lc and lc.fvs > 0.5:
        return "REVIEW — thesis at risk", RED

    strengthened = abs(c) > abs(c_prev)
    delta = abs(c) - abs(c_prev)

    if c > 0:  # Long position
        if mag >= 3.0 and strengthened:
            return "ADD — high conviction, strengthening", GREEN
        if mag >= 1.5 and strengthened:
            return "ADD — building conviction", GREEN
        if mag >= 1.5 and not strengthened:
            return "TRIM — conviction fading", RED
        if mag < 0.5 and c_prev != 0:
            return "EXIT CANDIDATE — low conviction", RED
        if mag < 0.5:
            return "WATCH — insufficient conviction", GRAY
        return "HOLD" if abs(delta) < 0.3 else ("ADD" if strengthened else "TRIM"), GRAY
    elif c < 0:  # Short position
        if mag >= 3.0 and strengthened:
            return "ADD SHORT — high conviction, strengthening", GREEN
        if mag >= 1.5 and strengthened:
            return "ADD SHORT — building conviction", GREEN
        if mag >= 1.5 and not strengthened:
            return "COVER — conviction fading", RED
        if mag < 0.5 and c_prev != 0:
            return "COVER CANDIDATE — low conviction", RED
        if mag < 0.5:
            return "WATCH — insufficient conviction", GRAY
        label = "HOLD SHORT" if abs(delta) < 0.3 else ("ADD SHORT" if strengthened else "COVER")
        return label, GRAY
    else:
        return "FLAT — no signal", GRAY


# ---------------------------------------------------------------------------
# Narrative generation
# ---------------------------------------------------------------------------
def generate_instrument_narrative(
    state: ConvictionState,
    raw_weight: float,
    constrained_weight: float,
    sizing_method: SizingMethod,
    failure_actions: list[FailureModeAction],
    params: ConvictionParams,
    n_positions: int = 1,
) -> str:
    """Build a structured, scannable narrative for one instrument.

    Returns markdown with:
      - Action signal (bold, one line)
      - Key drivers (bullet list)
      - Position detail (sizing info)
    """
    c = state.conviction
    c_prev = state.conviction_prev
    side = "Long" if c > 0 else ("Short" if c < 0 else "Flat")
    mag = abs(c)
    delta = c - c_prev

    # Conviction strength label
    if mag < 0.5:
        strength = "Low"
    elif mag < 2.0:
        strength = "Moderate"
    elif mag < 4.0:
        strength = "High"
    else:
        strength = "Near-max"

    # Action signal
    action, _ = _derive_action(state, failure_actions)

    lines: list[str] = []

    # Header line
    lines.append(
        f"**{state.instrument_id}** | {side} {c:+.2f} "
        f"(was {c_prev:+.2f}) | {strength} conviction"
    )
    lines.append(f"\n**Action: {action}**")

    # Key drivers
    drivers: list[str] = []
    lc = state.loss_components
    if lc:
        if abs(lc.fe) > 0.3:
            verb = "underperformed" if lc.fe > 0 else "outperformed"
            drivers.append(
                f"Forecast error: position {verb} expectations "
                f"(FE={lc.fe:+.2f})"
            )
        if lc.fvs > 0.2:
            drivers.append(
                f"Fundamental violation: thesis-relevant event "
                f"(FVS={lc.fvs:.2f})"
            )
        if abs(lc.rrs) > 0.05:
            direction = "increasing" if lc.rrs > 0 else "decreasing"
            drivers.append(
                f"Vol regime: idio vol {direction} "
                f"(RRS={lc.rrs:+.2f})"
            )
        if abs(lc.ads) > 0.03:
            drivers.append(
                f"IC debate shifted estimates "
                f"(ADS={lc.ads:+.2f})"
            )

    if not drivers and lc:
        components = {"FE": lc.fe, "FVS": lc.fvs, "RRS": lc.rrs, "ADS": lc.ads}
        dominant = max(components, key=lambda k: abs(components[k]))
        drivers.append(
            f"Primary driver: {dominant} ({components[dominant]:+.2f})"
        )

    if abs(delta) > 0.001:
        verb = "strengthened" if abs(c) > abs(c_prev) else "weakened"
        drivers.append(f"Conviction {verb} by {delta:+.2f} this period")

    # Failure mode alerts
    for fa in failure_actions:
        if fa.action_type == "halve_alpha":
            drivers.append(f"Oscillation guard: {fa.reason}")
        elif fa.action_type == "reset_conviction":
            drivers.append(f"Structural reset: {fa.reason}")

    if drivers:
        lines.append("\n**Drivers:**")
        for d in drivers:
            lines.append(f"- {d}")

    # Position sizing — skip the misleading raw weight for single positions
    method_labels = {
        SizingMethod.BASIC: "basic",
        SizingMethod.VOL_ADJUSTED: "vol-adjusted",
        SizingMethod.KELLY: "Kelly",
        SizingMethod.RISK_PARITY: "risk parity",
        SizingMethod.TIERED: "tiered",
    }
    method_name = method_labels.get(sizing_method, "basic")

    if n_positions == 1:
        lines.append(
            f"\n**Sizing ({method_name}):** "
            f"Constrained to {constrained_weight:+.4f} "
            f"(single position — raw weight is mechanically 1.0)"
        )
    else:
        lines.append(
            f"\n**Sizing ({method_name}):** "
            f"Raw {raw_weight:+.4f} → "
            f"Constrained {constrained_weight:+.4f}"
        )

    return "\n".join(lines)


def generate_portfolio_narrative(
    states: dict[str, ConvictionState],
    constrained: ConstrainedResult,
    constraints: PortfolioConstraints,
    failure_actions: dict[str, list[FailureModeAction]] | None = None,
) -> str:
    """Build a structured portfolio-level narrative."""
    weights = constrained.weights
    longs = {k: v for k, v in weights.items() if v > 0}
    shorts = {k: v for k, v in weights.items() if v < 0}

    gross = sum(abs(v) for v in weights.values())
    net = sum(weights.values())

    lines: list[str] = []

    # Exposure summary
    lines.append(
        f"**{len(longs)}** long, **{len(shorts)}** short "
        f"({len(weights)} positions) | "
        f"Gross {gross:.1%} | Net {net:+.1%}"
    )

    # Action items — positions that need attention
    if failure_actions:
        attention: list[str] = []
        for iid, actions in failure_actions.items():
            for fa in actions:
                if fa.action_type == "reset_conviction":
                    attention.append(f"**{iid}**: structural reset — review thesis")
                elif fa.action_type == "halve_alpha":
                    attention.append(f"**{iid}**: oscillation detected — damping updates")
        if attention:
            lines.append("\n**Attention required:**")
            for a in attention:
                lines.append(f"- {a}")

    # Constraint warnings
    if constrained.violations:
        lines.append("\n**Constraint violations:**")
        for v in constrained.violations:
            lines.append(f"- {v}")

    if not constrained.converged:
        lines.append(
            "\nNote: optimizer did not fully converge — "
            "constraints may be too tight for this portfolio."
        )

    # Largest positions
    if longs:
        largest_long = max(longs, key=longs.get)
        lines.append(
            f"\nLargest long: **{largest_long}** "
            f"({longs[largest_long]:+.4f})"
        )
    if shorts:
        largest_short = min(shorts, key=shorts.get)
        lines.append(
            f"Largest short: **{largest_short}** "
            f"({shorts[largest_short]:+.4f})"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------
def _build_print_report(
    states: dict[str, ConvictionState],
    raw_weights: dict[str, float],
    constrained: ConstrainedResult,
    instrument_narratives: dict[str, str],
    portfolio_narrative: str,
    params: ConvictionParams,
    constraints: PortfolioConstraints,
    sizing_method: SizingMethod,
) -> str:
    """Render an HTML report optimized for printing."""
    weights = constrained.weights
    gross = sum(abs(v) for v in weights.values())
    net = sum(weights.values())

    rows = ""
    for iid, st_obj in sorted(states.items()):
        lc = st_obj.loss_components
        side = (
            "Long" if st_obj.conviction > 0
            else "Short" if st_obj.conviction < 0
            else "Flat"
        )
        rows += f"""<tr>
            <td>{html.escape(iid)}</td>
            <td>{st_obj.conviction:+.3f}</td>
            <td>{raw_weights.get(iid, 0):+.4f}</td>
            <td>{weights.get(iid, 0):+.4f}</td>
            <td>{side}</td>
            <td>{f"{lc.fe:+.3f}" if lc else "—"}</td>
            <td>{f"{lc.fvs:.3f}" if lc else "—"}</td>
            <td>{f"{lc.rrs:+.3f}" if lc else "—"}</td>
            <td>{f"{lc.ads:+.3f}" if lc else "—"}</td>
            <td>{f"{lc.total_loss:.3f}" if lc else "—"}</td>
        </tr>"""

    narrative_sections = ""
    for iid in sorted(instrument_narratives):
        # Convert markdown narrative to simple HTML
        narr_html = instrument_narratives[iid]
        narr_html = narr_html.replace("**", "<b>", 1)
        # Replace remaining ** pairs with </b> and <b>
        while "**" in narr_html:
            narr_html = narr_html.replace("**", "</b>", 1)
            if "**" in narr_html:
                narr_html = narr_html.replace("**", "<b>", 1)
        narr_html = narr_html.replace("\n- ", "<br>• ")
        narr_html = narr_html.replace("\n", "<br>")
        narrative_sections += (
            f"<h3>{html.escape(iid)}</h3>"
            f"<div>{narr_html}</div>"
        )

    method_labels = {
        SizingMethod.BASIC: "Basic",
        SizingMethod.VOL_ADJUSTED: "Vol-Adjusted",
        SizingMethod.KELLY: "Kelly",
        SizingMethod.RISK_PARITY: "Risk Parity",
        SizingMethod.TIERED: "Tiered",
    }
    method_str = method_labels.get(sizing_method, "Basic")
    n_states = max(len(states), 1)
    avg_c = (
        sum(abs(s.conviction) for s in states.values()) / n_states
    )
    max_c = max(
        (abs(s.conviction) for s in states.values()), default=0
    )
    gen_date = date.today().isoformat()
    n_pos = len(weights)
    gross_pct = f"{gross:.1%}"
    net_pct = f"{net:+.1%}"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>CGF Portfolio Summary Report</title>
<style>
@media print {{{{
    .stApp, header, footer,
    [data-testid="stSidebar"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {{{{
        display: none !important;
    }}}}
    body {{{{
        margin: 0; padding: 20px;
        font-family: -apple-system, sans-serif;
    }}}}
}}}}
body {{{{
    font-family: -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
    max-width: 1100px; margin: 0 auto;
    padding: 24px; color: #222;
}}}}
h1 {{{{ border-bottom: 2px solid #333;
       padding-bottom: 8px; }}}}
h2 {{{{ color: #444; margin-top: 28px; }}}}
table {{{{
    border-collapse: collapse; width: 100%;
    margin: 12px 0; font-size: 13px;
}}}}
th, td {{{{ border: 1px solid #ccc;
           padding: 6px 10px; text-align: right; }}}}
th {{{{ background: #f5f5f5; font-weight: 600;
       text-align: center; }}}}
td:first-child {{{{ text-align: left;
                   font-weight: 500; }}}}
.kpi-row {{{{ display: flex; gap: 24px;
             margin: 16px 0; }}}}
.kpi {{{{
    background: #f8f8f8; border: 1px solid #ddd;
    border-radius: 6px; padding: 12px 18px;
    flex: 1; text-align: center;
}}}}
.kpi .label {{{{
    font-size: 12px; color: #888;
    text-transform: uppercase;
}}}}
.kpi .value {{{{
    font-size: 22px; font-weight: 600;
    margin-top: 4px;
}}}}
.params {{{{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px 20px; font-size: 13px;
    margin: 12px 0;
}}}}
.params span {{{{ color: #666; }}}}
p {{{{ line-height: 1.6; }}}}
</style></head><body>
<h1>CGF Portfolio Summary Report</h1>
<p style="color:#666;">
  Generated: {gen_date} | Sizing: {method_str}
</p>

<div class="kpi-row">
  <div class="kpi">
    <div class="label">Positions</div>
    <div class="value">{n_pos}</div>
  </div>
  <div class="kpi">
    <div class="label">Gross Exposure</div>
    <div class="value">{gross_pct}</div>
  </div>
  <div class="kpi">
    <div class="label">Net Exposure</div>
    <div class="value">{net_pct}</div>
  </div>
  <div class="kpi">
    <div class="label">Avg |C|</div>
    <div class="value">{avg_c:.3f}</div>
  </div>
  <div class="kpi">
    <div class="label">Max |C|</div>
    <div class="value">{max_c:.3f}</div>
  </div>
</div>

<h2>Position Table</h2>
<table>
<tr>
  <th>Ticker</th><th>C_t</th>
  <th>Raw Wt</th><th>Constrained Wt</th>
  <th>Side</th>
  <th>FE</th><th>FVS</th><th>RRS</th>
  <th>ADS</th><th>Total Loss</th>
</tr>
{rows}
</table>

<h2>Position Narratives</h2>
{narrative_sections}

<h2>Portfolio Summary</h2>
<p>{portfolio_narrative}</p>

<h2>Parameters</h2>
<div class="params">
  <div><span>w1:</span> {params.w1}</div>
  <div><span>w2:</span> {params.w2}</div>
  <div><span>w3:</span> {params.w3}</div>
  <div><span>w4:</span> {params.w4}</div>
  <div><span>kappa:</span> {params.kappa}</div>
  <div><span>alpha_min:</span> {params.alpha_min}</div>
  <div><span>alpha_max:</span> {params.alpha_max}</div>
  <div><span>beta:</span> {params.beta}</div>
  <div><span>C_max:</span> {params.C_max}</div>
</div>
<h3>Constraints</h3>
<div class="params">
  <div><span>Max gross:</span> {constraints.max_gross_exposure}</div>
  <div><span>Max net:</span> {constraints.max_net_exposure}</div>
  <div><span>Max position:</span> {constraints.max_position_pct}</div>
  <div><span>Max sector net:</span> {constraints.max_sector_net}</div>
</div>
</body></html>"""


# ---------------------------------------------------------------------------
# Portfolio loader
# ---------------------------------------------------------------------------
def _load_portfolio(vol_method_label: str):
    """Render portfolio loader section: paste tickers, upload CSV."""
    with st.expander("Batch — paste tickers or upload CSV", expanded=False):
        default_c = st.number_input(
            "Default Starting Conviction",
            value=0.0,
            min_value=-5.0,
            max_value=5.0,
            step=0.5,
            format="%.1f",
            help="Starting conviction for all batch-loaded tickers. "
            "Positive = long, negative = short. "
            "Set to 0 for new positions with no prior view.",
            key="batch_default_c",
        )

        loader_tab1, loader_tab2 = st.tabs(["Paste Tickers", "Upload CSV"])

        with loader_tab1:
            tickers_text = st.text_area(
                "Enter tickers (comma or newline separated)",
                placeholder="AAPL, TSLA, JNJ, PFE, JPM",
                height=100,
                key="portfolio_paste",
            )
            if st.button("Fetch All", key="fetch_paste") and tickers_text.strip():
                _process_ticker_list(
                    tickers_text.replace(",", "\n").split("\n"),
                    vol_method_label,
                    default_conviction=default_c,
                )

        with loader_tab2:
            uploaded = st.file_uploader(
                "CSV file",
                type=["csv"],
                help="Column `ticker` required. Optional: "
                "`price_target`, `conviction`.",
                key="portfolio_csv",
            )
            if uploaded is not None and st.button("Process CSV", key="fetch_csv"):
                import csv

                content = uploaded.read().decode("utf-8")
                reader = csv.DictReader(io.StringIO(content))
                rows = list(reader)
                if not rows:
                    st.warning("CSV is empty.")
                else:
                    # Normalize column names
                    cols_lower = {
                        c.strip().lower(): c for c in rows[0]
                    }
                    ticker_col = cols_lower.get("ticker")
                    target_col = cols_lower.get("price_target")
                    conv_col = cols_lower.get("conviction")

                    if ticker_col is None:
                        st.error("CSV must have a `ticker` column.")
                    else:
                        tickers = [
                            r[ticker_col].strip().upper()
                            for r in rows
                            if r[ticker_col].strip()
                        ]
                        targets = {}
                        convictions = {}
                        if target_col:
                            import contextlib

                            for r in rows:
                                tk = r[ticker_col].strip().upper()
                                with contextlib.suppress(ValueError, TypeError):
                                    targets[tk] = float(r[target_col])
                        if conv_col:
                            import contextlib

                            for r in rows:
                                tk = r[ticker_col].strip().upper()
                                with contextlib.suppress(ValueError, TypeError):
                                    convictions[tk] = float(r[conv_col])
                        _process_ticker_list(
                            tickers,
                            vol_method_label,
                            price_targets=targets,
                            default_conviction=default_c,
                            conviction_overrides=convictions,
                        )


def _process_ticker_list(
    raw_tickers: list[str],
    vol_method_label: str,
    price_targets: dict[str, float] | None = None,
    default_conviction: float = 0.0,
    conviction_overrides: dict[str, float] | None = None,
):
    """Fetch data for a list of tickers and add to instrument queue."""
    tickers = [t.strip().upper() for t in raw_tickers if t.strip()]
    if not tickers:
        st.warning("No tickers provided.")
        return

    progress = st.progress(0, text="Fetching market data...")
    successes = []
    failures = []

    for i, ticker in enumerate(tickers):
        progress.progress(
            (i + 1) / len(tickers),
            text=f"Fetching {ticker} ({i + 1}/{len(tickers)})...",
        )
        mkt = _fetch_instrument_data(ticker, vol_method_label)
        if mkt is None:
            failures.append(ticker)
            continue

        # Determine expected return from price target
        if price_targets and ticker in price_targets:
            target_px = price_targets[ticker]
        elif mkt.get("target_price"):
            target_px = mkt["target_price"]
        else:
            target_px = None

        current_px = mkt.get("current_price")
        if target_px and current_px and current_px > 0:
            expected_return = (target_px / current_px) - 1
        else:
            expected_return = 0.08  # fallback default

        # Per-ticker conviction from CSV override, else default
        c0 = (
            conviction_overrides.get(ticker, default_conviction)
            if conviction_overrides
            else default_conviction
        )

        inst_data = {
            "ticker": ticker,
            "as_of_date": date.today(),
            "realized_return": mkt["realized_return"],
            "expected_return": round(expected_return, 6),
            "sigma_expected": mkt["sigma_expected"],
            "sigma_idio_current": mkt["sigma_idio_current"],
            "sigma_idio_prev": mkt["sigma_idio_prev"],
            "sector": mkt.get("sector"),
            "implied_vol": mkt.get("implied_vol"),
            "historical_vol": mkt.get("historical_vol"),
            "target_price": target_px,
            "current_price": current_px,
            "initial_conviction": c0,
            "fvs_events": [],
            "p_pre": [],
            "p_post": [],
        }
        st.session_state.instruments.append(inst_data)
        successes.append(ticker)

    progress.empty()
    st.session_state.run_complete = False

    if successes:
        st.success(f"Added {len(successes)} instruments: {', '.join(successes)}")
    if failures:
        st.warning(f"Failed to fetch: {', '.join(failures)}")


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
def _run_pipeline(
    sizing_method: SizingMethod,
    params: ConvictionParams,
    constraints: PortfolioConstraints,
):
    instruments = st.session_state.instruments
    if not instruments:
        st.error("Add at least one instrument before running.")
        return

    data_batch: list[InstrumentData] = []
    sectors: dict[str, str] = {}
    expected_returns_map: dict[str, float] = {}
    for inst in instruments:
        fvs_events = []
        for ev in inst.get("fvs_events", []):
            fvs_events.append({
                "event_type": ev["event_type"],
                "event_date": ev["event_date"],
                "description": ev.get("description", ""),
            })

        p_pre = inst.get("p_pre", [])
        p_post = inst.get("p_post", [])

        data_batch.append(InstrumentData(
            instrument_id=inst["ticker"],
            as_of_date=inst["as_of_date"],
            realized_return=inst["realized_return"],
            expected_return=inst["expected_return"],
            sigma_expected=inst["sigma_expected"],
            sigma_idio_current=inst["sigma_idio_current"],
            sigma_idio_prev=inst["sigma_idio_prev"],
            implied_vol=inst.get("implied_vol"),
            historical_vol=inst.get("historical_vol"),
            fvs_events=fvs_events,
            p_pre=p_pre,
            p_post=p_post,
        ))
        if inst.get("sector"):
            sectors[inst["ticker"]] = inst["sector"]
        expected_returns_map[inst["ticker"]] = inst["expected_return"]

    # Seed conviction states with user-provided initial convictions
    states: dict[str, ConvictionState] = {}
    for inst in instruments:
        c0 = inst.get("initial_conviction", 0.0)
        if c0 != 0.0:
            states[inst["ticker"]] = ConvictionState(
                instrument_id=inst["ticker"],
                as_of_date=inst["as_of_date"],
                conviction=c0,
                conviction_prev=c0,
            )

    updated_states = run_batch_update(
        states, data_batch, params, TAXONOMY
    )

    failure_actions: dict[str, list[FailureModeAction]] = {
        k: [] for k in updated_states
    }
    for iid, state in updated_states.items():
        osc = check_oscillation_guard(
            iid,
            [state.conviction_prev, state.conviction],
            state.alpha_t,
            params,
        )
        if osc:
            failure_actions[iid].append(osc)

        lc = state.loss_components
        inst_data = next(
            d for d in data_batch if d.instrument_id == iid
        )
        if lc:
            sr = check_structural_reset(
                iid, lc.fvs,
                inst_data.sigma_idio_current,
                inst_data.sigma_idio_prev, params,
            )
            if sr:
                failure_actions[iid].append(sr)

    convictions = {
        iid: s.conviction for iid, s in updated_states.items()
    }
    vols = {
        iid: s.idiosyncratic_vol
        for iid, s in updated_states.items()
    }

    # Determine which extra args the sizing method needs
    needs_vols = sizing_method in (
        SizingMethod.VOL_ADJUSTED,
        SizingMethod.KELLY,
        SizingMethod.RISK_PARITY,
    )
    use_vols = vols if needs_vols else None
    use_er = (
        expected_returns_map
        if sizing_method == SizingMethod.KELLY
        else None
    )
    raw_weights = map_convictions(
        convictions, sizing_method, use_vols, use_er
    )

    constrained = apply_constraints(
        raw_weights, constraints, sectors if sectors else None
    )

    n_positions = len(updated_states)
    instrument_narratives: dict[str, str] = {}
    for iid, state in updated_states.items():
        instrument_narratives[iid] = generate_instrument_narrative(
            state,
            raw_weights.get(iid, 0.0),
            constrained.weights.get(iid, 0.0),
            sizing_method,
            failure_actions.get(iid, []),
            params,
            n_positions=n_positions,
        )
    portfolio_narrative = generate_portfolio_narrative(
        updated_states, constrained, constraints, failure_actions
    )

    st.session_state.results = {
        "states": updated_states,
        "raw_weights": raw_weights,
        "constrained": constrained,
        "instrument_narratives": instrument_narratives,
        "portfolio_narrative": portfolio_narrative,
        "failure_actions": failure_actions,
        "sizing_method": sizing_method,
        "params": params,
        "constraints": constraints,
        "sectors": sectors,
    }
    st.session_state.run_complete = True


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
def main():  # noqa: C901
    sizing_method, vol_method_label, params, constraints = _sidebar()

    # --- Header ---
    col_title, col_glossary = st.columns([5, 1])
    with col_title:
        st.title("Conviction Gradient Framework")
        st.caption(
            "Gradient-descent conviction engine for L/S portfolio sizing"
        )
    with col_glossary:
        st.markdown("")  # vertical spacer
        st.markdown("")
        if st.button("Glossary"):
            _show_glossary()

    # --- Instructions ---
    st.info(
        "**Quick start:** Enter a ticker — market data and consensus "
        "price targets are fetched automatically. Adjust the price "
        "target if you have a different view. Queue multiple "
        "instruments (or use **Load Portfolio** for batch entry), "
        "then click **Run Conviction Update**. Results appear in "
        "tabs below. Use the sidebar to adjust engine parameters, "
        "vol method, and sizing method.",
        icon="📋",
    )

    # --- How This Works ---
    with st.expander("How This Works"):
        st.markdown(
            "**What CGF does:** For each stock in your portfolio, "
            "CGF computes a *conviction score* — a signed number "
            "reflecting how strongly the evidence supports the "
            "position. Each period, conviction updates via "
            "gradient descent: it moves in the direction that "
            "reduces a loss function built from four signals "
            "(forecast error, fundamental violations, risk regime "
            "shifts, and IC debate shifts). Conviction then maps "
            "to portfolio weights subject to risk constraints."
        )
        st.markdown(
            "**Where data comes from:** When you enter a ticker, "
            "the app fetches ~1 year of price history from "
            "**Yahoo Finance** (via yfinance). From that data it "
            "computes: trailing realized return, historical "
            "volatility, and idiosyncratic volatility (stock-"
            "specific risk after stripping out market moves via "
            "CAPM regression against SPY). Consensus price targets "
            "are auto-populated from analyst estimates; your "
            "expected return is computed as "
            "(price_target / current_price) - 1."
        )
        st.markdown(
            "**Optional inputs:** You can also log Fundamental "
            "Violation events (management changes, guidance cuts, "
            "governance issues) and IC Debate probability shifts "
            "to incorporate qualitative signals into the "
            "conviction update."
        )
        st.markdown(
            "**Conviction to weights:** Convictions are mapped "
            "to portfolio weights using one of five sizing methods "
            "(basic, vol-adjusted, Kelly, risk parity, or tiered), "
            "then constrained to enforce gross/net exposure, "
            "position concentration, and sector limits."
        )
        st.markdown(
            "**Inspiration:** Traditional discretionary portfolio "
            "management treats conviction as a gut-feel number — "
            "a PM says \"I'm high conviction on X\" without a "
            "systematic framework for how that conviction should "
            "update when new information arrives. CGF borrows the "
            "core idea from machine learning: treat conviction as "
            "a parameter being optimized via gradient descent, "
            "where the \"loss function\" measures how wrong the "
            "current conviction appears to be given realized "
            "evidence (forecast misses, thesis violations, vol "
            "regime changes, debate outcomes). The result is a "
            "conviction updating process that is transparent, "
            "auditable, and consistent — the same evidence "
            "produces the same conviction change every time, "
            "removing the behavioral biases (anchoring, loss "
            "aversion, recency bias) that tend to degrade "
            "portfolio returns over time."
        )

    # --- Portfolio Loader ---
    st.header("Add Instruments")
    _load_portfolio(vol_method_label)

    # --- Scenario Input ---
    st.subheader("Single Instrument")

    with st.form("instrument_form", clear_on_submit=True):
        st.subheader("Instrument Data")
        cols = st.columns([1.5, 1.5, 1, 1])
        ticker = cols[0].text_input(
            "Ticker", placeholder="AAPL"
        )
        price_target_input = cols[1].number_input(
            "Price Target ($)",
            value=0.0,
            min_value=0.0,
            format="%.2f",
            help="Leave at 0 to auto-populate from consensus.",
        )
        initial_conviction = cols[2].number_input(
            "Starting C",
            value=0.0,
            min_value=-5.0,
            max_value=5.0,
            step=0.5,
            format="%.1f",
            help="Your current conviction. Positive = long, negative = short. "
            "Set to 0 for a new position with no prior view.",
        )
        as_of_date = cols[3].date_input(
            "As-of Date", value=date.today()
        )

        # FVS Events
        with st.expander("FVS Events (optional)"):
            fvs_cols = st.columns([1, 1, 2])
            fvs_event_type = fvs_cols[0].selectbox(
                "Event Type",
                options=TAXONOMY.event_types,
                index=0,
            )
            fvs_event_date = fvs_cols[1].date_input(
                "Event Date", value=date.today(),
                key="fvs_date",
            )
            fvs_description = fvs_cols[2].text_input(
                "Description", key="fvs_desc"
            )
            add_fvs = st.checkbox(
                "Include this FVS event", value=False
            )

        # IC Debate
        with st.expander("IC Debate (optional)"):
            n_participants = st.number_input(
                "Number of IC participants",
                min_value=0, max_value=5, value=0, step=1,
            )
            p_pre_list: list[float] = []
            p_post_list: list[float] = []
            if n_participants > 0:
                debate_cols = st.columns(int(n_participants))
                for i in range(int(n_participants)):
                    with debate_cols[i]:
                        st.markdown(f"**P{i + 1}**")
                        pp = st.number_input(
                            "p_pre", 0.0, 1.0, 0.5, 0.05,
                            key=f"ppre_{i}", format="%.2f",
                        )
                        po = st.number_input(
                            "p_post", 0.0, 1.0, 0.5, 0.05,
                            key=f"ppost_{i}", format="%.2f",
                        )
                        p_pre_list.append(pp)
                        p_post_list.append(po)

        # Override market data
        with st.expander("Override Market Data (advanced)"):
            st.caption(
                "Leave at 0 to use auto-fetched values. "
                "Set a non-zero value to override."
            )
            ov_cols = st.columns(3)
            ov_realized = ov_cols[0].number_input(
                "Realized Return", value=0.0,
                format="%.6f", key="ov_realized",
            )
            ov_sigma_exp = ov_cols[1].number_input(
                "Sigma Expected", value=0.0,
                min_value=0.0, format="%.6f",
                key="ov_sigma_exp",
            )
            ov_sigma_idio_cur = ov_cols[2].number_input(
                "Sigma Idio (current)", value=0.0,
                min_value=0.0, format="%.6f",
                key="ov_sigma_idio_cur",
            )
            ov_cols2 = st.columns(3)
            ov_sigma_idio_prev = ov_cols2[0].number_input(
                "Sigma Idio (prev)", value=0.0,
                min_value=0.0, format="%.6f",
                key="ov_sigma_idio_prev",
            )
            ov_hist_vol = ov_cols2[1].number_input(
                "Historical Vol", value=0.0,
                min_value=0.0, format="%.6f",
                key="ov_hist_vol",
            )
            ov_sector = ov_cols2[2].text_input(
                "Sector", value="", key="ov_sector",
            )

        submitted = st.form_submit_button(
            "Add Instrument", type="secondary"
        )

    # --- Handle form submission ---
    if submitted and ticker.strip():
        ticker_clean = ticker.strip().upper()

        # Check for overrides first
        has_overrides = (
            ov_realized != 0.0
            or ov_sigma_exp > 0.0
            or ov_sigma_idio_cur > 0.0
        )

        if has_overrides:
            # Use overrides — require sigma fields > 0
            if ov_sigma_exp <= 0 or ov_sigma_idio_cur <= 0:
                st.error(
                    "When overriding, sigma_expected and "
                    "sigma_idio_current must be > 0."
                )
            else:
                mkt = {
                    "realized_return": ov_realized,
                    "sigma_expected": ov_sigma_exp,
                    "sigma_idio_current": ov_sigma_idio_cur,
                    "sigma_idio_prev": (
                        ov_sigma_idio_prev
                        if ov_sigma_idio_prev > 0
                        else ov_sigma_idio_cur
                    ),
                    "historical_vol": (
                        ov_hist_vol if ov_hist_vol > 0 else None
                    ),
                    "sector": (
                        ov_sector.strip()
                        if ov_sector.strip()
                        else None
                    ),
                    "target_price": None,
                    "current_price": None,
                    "n_analysts": 0,
                }
                fetch_ok = True
        else:
            # Auto-fetch
            mkt = _fetch_instrument_data(ticker_clean, vol_method_label)
            fetch_ok = mkt is not None

        if not fetch_ok:
            st.error(
                f"Could not fetch market data for "
                f"**{ticker_clean}**. Check the ticker or "
                f"use the Override Market Data section."
            )
        else:
            # Determine price target and expected return
            if price_target_input > 0:
                target_px = price_target_input
            elif mkt.get("target_price"):
                target_px = mkt["target_price"]
            else:
                target_px = None

            current_px = mkt.get("current_price")
            if target_px and current_px and current_px > 0:
                expected_return = (target_px / current_px) - 1
            else:
                expected_return = 0.08  # fallback

            inst_data = {
                "ticker": ticker_clean,
                "as_of_date": as_of_date,
                "realized_return": mkt["realized_return"],
                "expected_return": round(expected_return, 6),
                "sigma_expected": mkt["sigma_expected"],
                "sigma_idio_current": mkt["sigma_idio_current"],
                "sigma_idio_prev": mkt["sigma_idio_prev"],
                "sector": mkt.get("sector"),
                "implied_vol": mkt.get("implied_vol"),
                "historical_vol": mkt.get("historical_vol"),
                "target_price": target_px,
                "current_price": current_px,
                "initial_conviction": initial_conviction,
                "fvs_events": [],
                "p_pre": p_pre_list,
                "p_post": p_post_list,
            }
            if add_fvs:
                inst_data["fvs_events"].append({
                    "event_type": fvs_event_type,
                    "event_date": fvs_event_date,
                    "description": fvs_description,
                })
            st.session_state.instruments.append(inst_data)
            st.session_state.run_complete = False

            # Show fetched data confirmation with consensus info
            consensus_info = ""
            if mkt.get("target_price") and mkt.get("current_price"):
                implied_er = (mkt["target_price"] / mkt["current_price"]) - 1
                consensus_info = (
                    f" | Consensus: ${mkt['target_price']:.2f} "
                    f"({mkt.get('n_analysts', 0)} analysts) "
                    f"| Current: ${mkt['current_price']:.2f} "
                    f"| Implied E[R]: {implied_er:+.1%}"
                )

            st.success(
                f"Added **{ticker_clean}** — "
                f"E[R]: {expected_return:.2%}, "
                f"Sigma Exp: {mkt['sigma_expected']:.2%}, "
                f"Sigma Idio: "
                f"{mkt['sigma_idio_current']:.2%}, "
                f"Sector: {mkt.get('sector', 'N/A')}"
                f"{consensus_info}"
            )

    # --- Instrument queue ---
    if st.session_state.instruments:
        st.subheader(
            f"Instrument Queue "
            f"({len(st.session_state.instruments)})"
        )
        queue_data = []
        for inst in st.session_state.instruments:
            target_str = (
                f"${inst['target_price']:.2f}"
                if inst.get("target_price")
                else "—"
            )
            queue_data.append({
                "Ticker": inst["ticker"],
                "C_0": f"{inst.get('initial_conviction', 0.0):+.1f}",
                "Date": str(inst["as_of_date"]),
                "Target": target_str,
                "E[R]": f"{inst['expected_return']:.2%}",
                "Realized": f"{inst['realized_return']:.2%}",
                "Sigma Idio": f"{inst['sigma_idio_current']:.2%}",
                "Sector": inst.get("sector") or "—",
            })
        st.dataframe(
            queue_data, use_container_width=True, hide_index=True
        )

        col_run, col_clear = st.columns([1, 1])
        with col_run:
            if st.button(
                "Run Conviction Update",
                type="primary",
                use_container_width=True,
            ):
                _run_pipeline(sizing_method, params, constraints)
        with col_clear:
            if st.button("Clear All", use_container_width=True):
                st.session_state.instruments = []
                st.session_state.run_complete = False
                st.rerun()

    # --- Results Section ---
    if st.session_state.run_complete:
        res = st.session_state.results
        states = res["states"]
        raw_weights = res["raw_weights"]
        constrained: ConstrainedResult = res["constrained"]
        instrument_narratives = res["instrument_narratives"]
        portfolio_narrative = res["portfolio_narrative"]

        st.divider()
        st.header("Results")

        weights = constrained.weights
        gross = sum(abs(v) for v in weights.values())
        net_exp = sum(weights.values())
        abs_convictions = [
            abs(s.conviction) for s in states.values()
        ]
        avg_c = (
            sum(abs_convictions) / len(abs_convictions)
            if abs_convictions
            else 0
        )
        max_c = max(abs_convictions) if abs_convictions else 0

        kpi_cols = st.columns(5)
        kpi_cols[0].metric("Positions", len(weights))
        kpi_cols[1].metric("Gross Exposure", f"{gross:.1%}")
        kpi_cols[2].metric("Net Exposure", f"{net_exp:+.1%}")
        kpi_cols[3].metric("Avg |C|", f"{avg_c:.3f}")
        kpi_cols[4].metric("Max |C|", f"{max_c:.3f}")

        # Action summary — the most actionable view
        failure_acts = res.get("failure_actions", {})
        action_data = []
        for iid, state in sorted(states.items()):
            action_label, _ = _derive_action(
                state, failure_acts.get(iid, [])
            )
            action_data.append({
                "Ticker": iid,
                "Action": action_label.split(" — ")[0],
                "Reason": action_label.split(" — ")[1] if " — " in action_label else "",
                "C_t": f"{state.conviction:+.2f}",
                "Delta": f"{state.conviction - state.conviction_prev:+.2f}",
                "Weight": f"{weights.get(iid, 0):+.4f}",
            })
        st.dataframe(
            action_data, use_container_width=True, hide_index=True
        )

        tab1, tab2, tab3, tab4 = st.tabs([
            "Conviction States", "Portfolio Weights",
            "Charts", "Narrative Analysis",
        ])

        with tab1:
            conv_data = []
            for iid, state in sorted(states.items()):
                lc = state.loss_components
                gr = state.gradient
                conv_data.append({
                    "Ticker": iid,
                    "C_t": f"{state.conviction:+.3f}",
                    "C_prev": (
                        f"{state.conviction_prev:+.3f}"
                    ),
                    "Delta": (
                        f"{state.conviction - state.conviction_prev:+.3f}"
                    ),
                    "alpha_t": f"{state.alpha_t:.4f}",
                    "FE": (
                        f"{lc.fe:+.3f}" if lc else "—"
                    ),
                    "FVS": (
                        f"{lc.fvs:.3f}" if lc else "—"
                    ),
                    "RRS": (
                        f"{lc.rrs:+.3f}" if lc else "—"
                    ),
                    "ADS": (
                        f"{lc.ads:+.3f}" if lc else "—"
                    ),
                    "Total Loss": (
                        f"{lc.total_loss:.3f}" if lc else "—"
                    ),
                    "Gradient": (
                        f"{gr.gradient_value:+.4f}"
                        if gr else "—"
                    ),
                })
            st.dataframe(
                conv_data,
                use_container_width=True,
                hide_index=True,
            )

        with tab2:
            weight_data = []
            for iid in sorted(weights):
                rw = raw_weights.get(iid, 0)
                cw = weights[iid]
                weight_data.append({
                    "Ticker": iid,
                    "Raw Weight": f"{rw:+.4f}",
                    "Constrained Weight": f"{cw:+.4f}",
                    "Side": (
                        "Long" if cw > 0
                        else ("Short" if cw < 0 else "Flat")
                    ),
                })
            st.dataframe(
                weight_data,
                use_container_width=True,
                hide_index=True,
            )

            tickers = sorted(weights.keys())
            fig_w = go.Figure()
            fig_w.add_trace(go.Bar(
                y=tickers,
                x=[raw_weights.get(t, 0) for t in tickers],
                name="Raw",
                orientation="h",
                marker_color=GRAY,
                opacity=0.6,
            ))
            fig_w.add_trace(go.Bar(
                y=tickers,
                x=[weights[t] for t in tickers],
                name="Constrained",
                orientation="h",
                marker_color=[
                    GREEN if weights[t] > 0 else RED
                    for t in tickers
                ],
            ))
            fig_w.update_layout(
                template=PLOTLY_TEMPLATE,
                title="Raw vs Constrained Weights",
                xaxis_title="Weight",
                barmode="group",
                height=max(300, 60 * len(tickers)),
                margin=dict(l=10, r=10, t=40, b=30),
            )
            st.plotly_chart(fig_w, use_container_width=True)

        with tab3:
            tickers_sorted = sorted(states.keys())
            convictions = [
                states[t].conviction for t in tickers_sorted
            ]
            colors = [
                GREEN if c > 0 else RED for c in convictions
            ]

            fig_c = go.Figure(go.Bar(
                x=tickers_sorted,
                y=convictions,
                marker_color=colors,
                text=[f"{c:+.2f}" for c in convictions],
                textposition="outside",
            ))
            fig_c.update_layout(
                template=PLOTLY_TEMPLATE,
                title="Conviction Scores",
                yaxis_title="Conviction (C_t)",
                height=400,
                margin=dict(l=10, r=10, t=40, b=30),
            )
            st.plotly_chart(fig_c, use_container_width=True)

            fe_vals, fvs_vals = [], []
            rrs_vals, ads_vals = [], []
            for t in tickers_sorted:
                lc = states[t].loss_components
                fe_vals.append(abs(lc.fe) if lc else 0)
                fvs_vals.append(lc.fvs if lc else 0)
                rrs_vals.append(abs(lc.rrs) if lc else 0)
                ads_vals.append(abs(lc.ads) if lc else 0)

            fig_loss = go.Figure()
            for name, vals, color in [
                ("FE", fe_vals, "#3498db"),
                ("FVS", fvs_vals, "#e67e22"),
                ("RRS", rrs_vals, "#9b59b6"),
                ("ADS", ads_vals, "#1abc9c"),
            ]:
                fig_loss.add_trace(go.Bar(
                    name=name, x=tickers_sorted,
                    y=vals, marker_color=color,
                ))
            fig_loss.update_layout(
                template=PLOTLY_TEMPLATE,
                title="Loss Component Decomposition",
                yaxis_title="|Component Value|",
                barmode="stack",
                height=400,
                margin=dict(l=10, r=10, t=40, b=30),
            )
            st.plotly_chart(fig_loss, use_container_width=True)

            fig_alloc = go.Figure(go.Bar(
                x=tickers_sorted,
                y=[
                    weights.get(t, 0) for t in tickers_sorted
                ],
                marker_color=[
                    GREEN if weights.get(t, 0) > 0 else RED
                    for t in tickers_sorted
                ],
                text=[
                    f"{weights.get(t, 0):+.3f}"
                    for t in tickers_sorted
                ],
                textposition="outside",
            ))
            fig_alloc.update_layout(
                template=PLOTLY_TEMPLATE,
                title="Weight Allocation",
                yaxis_title="Constrained Weight",
                height=400,
                margin=dict(l=10, r=10, t=40, b=30),
            )
            st.plotly_chart(fig_alloc, use_container_width=True)

        with tab4:
            st.subheader("Per-Instrument Analysis")
            for iid in sorted(instrument_narratives):
                st.markdown(
                    f"---\n{instrument_narratives[iid]}"
                )

            st.subheader("Portfolio Summary")
            st.markdown(portfolio_narrative)

        # --- Print Report ---
        st.divider()
        report_html = _build_print_report(
            states,
            raw_weights,
            constrained,
            instrument_narratives,
            portfolio_narrative,
            res["params"],
            res["constraints"],
            res["sizing_method"],
        )

        if st.button(
            "Print Portfolio Summary Report", type="primary"
        ):
            st.components.v1.html(
                report_html + """
                <script>
                window.addEventListener('load', function() {
                    setTimeout(function() {
                        window.print();
                    }, 500);
                });
                </script>
                """,
                height=800,
                scrolling=True,
            )

        with st.expander("Preview Report HTML"):
            st.components.v1.html(
                report_html, height=600, scrolling=True
            )


if __name__ == "__main__":
    main()
else:
    main()
