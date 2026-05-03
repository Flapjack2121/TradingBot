"""Daily scheduled run: refresh data, scan signals, log them, notify.

Run as a long-lived process (e.g. inside a container or systemd unit):

    python -m scheduler.daily

Or as a one-shot:

    python -m scheduler.daily --once
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running as a script: ``python scheduler/daily.py``.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import schedule

import settings
from data import DataManager
from strategies import REGISTRY, SignalEngine
from strategies.engine import RiskConfig
from database import TradeRepo
from notifications import Notifier
from notifications.notifier import EmailConfig, TelegramConfig
from reports import Reporter

log = logging.getLogger("scheduler")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_once() -> None:
    cfg = settings.load_config()
    dm = DataManager(
        cache_dir=cfg.get("data", {}).get("cache_dir", "data/cache"),
        period=cfg.get("data", {}).get("history_period", "2y"),
        interval=cfg.get("data", {}).get("interval", "1d"),
        cache_ttl_hours=cfg.get("data", {}).get("cache_ttl_hours", 12),
        use_adjusted=cfg.get("data", {}).get("use_adjusted", True),
        benchmark=cfg.get("universe", {}).get("benchmark", "SPY"),
        indicator_cfg=cfg.get("indicators", {}),
    )

    universe_cfg = cfg.get("universe", {})
    tickers = dm.universe(
        source=universe_cfg.get("source", "sp500"),
        explicit=universe_cfg.get("tickers") or None,
        max_symbols=universe_cfg.get("max_symbols", 100),
    )
    log.info("Scanning %d tickers", len(tickers))

    risk_cfg = cfg.get("risk", {})
    strat_cfg = cfg.get("strategies", {})
    enabled = strat_cfg.get("enabled", list(REGISTRY.keys()))
    strategies = [REGISTRY[name](strat_cfg.get(name, {})) for name in enabled if name in REGISTRY]
    engine = SignalEngine(
        strategies=strategies,
        risk=RiskConfig(
            account_size=risk_cfg.get("account_size", 10000),
            risk_per_trade=risk_cfg.get("risk_per_trade", 0.01),
            atr_stop_multiplier=risk_cfg.get("atr_stop_multiplier", 2.0),
            take_profit_r_multiple=risk_cfg.get("take_profit_r_multiple", 2.0),
        ),
    )

    frames = dm.bulk(tickers, force_refresh=True)
    signals = engine.scan(frames)
    actionable = engine.actionable(signals)
    log.info("Total signals=%d, actionable=%d", len(signals), len(actionable))

    repo = TradeRepo(cfg.get("database", {}).get("path", "database/trades.db"))
    repo.log_signals(signals)

    Reporter().save_signals(signals, fmt="json")

    notif = cfg.get("notifications", {})
    if notif.get("enabled") and notif.get("channels") and actionable:
        notifier = Notifier(
            channels=notif.get("channels", []),
            email=EmailConfig(**(notif.get("email") or {})),
            telegram=TelegramConfig(**(notif.get("telegram") or {})),
        )
        notifier.send_signals(actionable, header="Daily trading signals")
        log.info("Sent %d signals via %s", len(actionable), notif.get("channels"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run a single scan then exit")
    parser.add_argument("--time", default="22:30", help="HH:MM (local) for daily run")
    args = parser.parse_args()

    if args.once:
        run_once()
        return

    schedule.every().day.at(args.time).do(run_once)
    log.info("Scheduler armed for %s daily.", args.time)
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
