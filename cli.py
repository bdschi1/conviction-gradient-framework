"""CLI entry point for the Conviction Gradient Framework."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date


def cmd_update(args: argparse.Namespace) -> None:
    """Run a conviction update from a JSON data file."""
    from engine.models import ConvictionState, InstrumentData
    from engine.updater import run_batch_update
    from storage.database import ConvictionDB

    with open(args.data_file) as f:
        raw = json.load(f)

    data_batch = [InstrumentData(**d) for d in raw]
    db = ConvictionDB(args.db)

    # Load current states
    states = {}
    for d in data_batch:
        latest = db.get_latest_state(d.instrument_id)
        if latest:
            states[d.instrument_id] = ConvictionState(
                instrument_id=d.instrument_id,
                as_of_date=date.fromisoformat(latest["as_of_date"]),
                conviction=latest["conviction"],
                conviction_prev=latest["conviction_prev"],
            )

    updated = run_batch_update(states, data_batch)

    for state in updated.values():
        db.save_state(state)
        print(
            f"  {state.instrument_id}: C={state.conviction:.4f} "
            f"(prev={state.conviction_prev:.4f}, alpha={state.alpha_t:.4f})"
        )

    print(f"\nUpdated {len(updated)} instruments.")
    db.close()


def cmd_trajectory(args: argparse.Namespace) -> None:
    """Print conviction trajectory for an instrument."""
    from storage.database import ConvictionDB

    db = ConvictionDB(args.db)
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None
    trajectory = db.get_trajectory(args.instrument, start, end)

    if not trajectory:
        print(f"No data for {args.instrument}")
        return

    print(f"\nConviction trajectory for {args.instrument}:")
    print(f"{'Date':<12} {'C_t':>8} {'Alpha':>8} {'Loss':>8} {'Gradient':>10}")
    print("-" * 50)
    for row in trajectory:
        print(
            f"{row['as_of_date']:<12} "
            f"{row['conviction']:>8.4f} "
            f"{row.get('alpha_t', 0):>8.4f} "
            f"{row.get('total_loss', 0) or 0:>8.4f} "
            f"{row.get('gradient_value', 0) or 0:>10.4f}"
        )
    db.close()


def cmd_validate(args: argparse.Namespace) -> None:
    """Run validation harness over synthetic or real data."""
    from validation.report import (
        format_ablation_summary,
        format_comparison_table,
        format_multi_seed_summary,
        format_real_data_summary,
        format_signal_summary,
    )
    from validation.runner import (
        run_real_data_validation,
        run_signal_validation_multi_seed,
        run_validation_multi_seed,
    )

    n_seeds = args.seeds
    n_days = args.days

    if args.real:
        from validation.real_data import RealDataConfig

        tickers = args.tickers.split(",") if args.tickers else None
        config = RealDataConfig(
            tickers=tickers or RealDataConfig().tickers,
            start_date=args.start or RealDataConfig().start_date,
            end_date=args.end or RealDataConfig().end_date,
        )
        print(f"Running real data validation ({', '.join(config.tickers)})...")
        print(f"Period: {config.start_date} to {config.end_date}\n")
        result = run_real_data_validation(config)
        print(format_comparison_table(result))
        print()
        print(format_real_data_summary(result, tickers=config.tickers))
    elif args.signal:
        print(f"Running signal-embedded validation ({n_seeds} seeds, {n_days} days)...\n")
        results = run_signal_validation_multi_seed(
            n_seeds=n_seeds, n_days=n_days,
        )
        print(format_comparison_table(results[0]))
        print()
        print(format_ablation_summary(results[0]))
        print()
        print(format_signal_summary(results))
    else:
        print(f"Running validation ({n_seeds} seeds, {n_days} days each)...\n")
        results = run_validation_multi_seed(n_seeds=n_seeds, n_days=n_days)
        print(format_comparison_table(results[0]))
        print()
        print(format_ablation_summary(results[0]))
        print()
        print(format_multi_seed_summary(results))


def cmd_health(args: argparse.Namespace) -> None:
    """Show database health stats."""
    from storage.database import ConvictionDB

    db = ConvictionDB(args.db)
    conn = db._get_conn()
    try:
        n_states = conn.execute("SELECT COUNT(*) FROM conviction_states").fetchone()[0]
        n_instruments = conn.execute(
            "SELECT COUNT(DISTINCT instrument_id) FROM conviction_states"
        ).fetchone()[0]
        n_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        n_sessions = conn.execute("SELECT COUNT(*) FROM ic_sessions").fetchone()[0]

        print(f"Database: {args.db}")
        print(f"  Conviction states: {n_states}")
        print(f"  Instruments: {n_instruments}")
        print(f"  Events: {n_events}")
        print(f"  IC sessions: {n_sessions}")
    except Exception as e:
        print(f"Database error: {e}")
    finally:
        db.close()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="cgf",
        description="Conviction Gradient Framework",
    )
    parser.add_argument("--db", default="conviction.db", help="SQLite database path")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # update
    p_update = subparsers.add_parser("update", help="Run conviction update from data file")
    p_update.add_argument("data_file", help="JSON file with instrument data")

    # trajectory
    p_traj = subparsers.add_parser("trajectory", help="Show conviction trajectory")
    p_traj.add_argument("instrument", help="Instrument ID (ticker)")
    p_traj.add_argument("--start", help="Start date (YYYY-MM-DD)")
    p_traj.add_argument("--end", help="End date (YYYY-MM-DD)")

    # validate
    p_validate = subparsers.add_parser("validate", help="Run validation harness")
    p_validate.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    p_validate.add_argument("--days", type=int, default=504, help="Trading days per seed")
    p_validate.add_argument(
        "--signal", action="store_true",
        help="Use signal-embedded synthetic data (tests component value)",
    )
    p_validate.add_argument(
        "--real", action="store_true",
        help="Use real market data via yfinance (requires network)",
    )
    p_validate.add_argument(
        "--tickers", type=str, default=None,
        help="Comma-separated tickers for --real (default: AAPL,MSFT,JPM,XOM,JNJ)",
    )
    p_validate.add_argument(
        "--start", type=str, default=None,
        help="Start date for --real (YYYY-MM-DD, default: 2022-01-01)",
    )
    p_validate.add_argument(
        "--end", type=str, default=None,
        help="End date for --real (YYYY-MM-DD, default: 2023-12-31)",
    )

    # health
    subparsers.add_parser("health", help="Show database health")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s - %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.command == "update":
        cmd_update(args)
    elif args.command == "trajectory":
        cmd_trajectory(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "health":
        cmd_health(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
