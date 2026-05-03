"""Performance and signal-history exporters."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


class Reporter:
    def __init__(self, out_dir: str | Path = "reports/output") -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _stamp(self) -> str:
        return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_signals(self, signals: Iterable, fmt: str = "csv") -> Path:
        rows = [s.to_dict() for s in signals]
        df = pd.DataFrame(rows)
        path = self.out_dir / f"signals_{self._stamp()}.{fmt}"
        if fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "json":
            path.write_text(json.dumps(rows, indent=2, default=str))
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        return path

    def save_trades(self, trades_df: pd.DataFrame, fmt: str = "csv") -> Path:
        path = self.out_dir / f"trades_{self._stamp()}.{fmt}"
        if fmt == "csv":
            trades_df.to_csv(path, index=False)
        elif fmt == "json":
            trades_df.to_json(path, orient="records", indent=2, date_format="iso")
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        return path

    def save_backtest(self, name: str, result) -> Path:
        path = self.out_dir / f"backtest_{name}_{self._stamp()}.json"
        payload = {
            "stats": result.stats,
            "initial_cash": result.initial_cash,
            "final_equity": result.final_equity,
            "total_return_pct": result.total_return_pct,
            "trades": result.trades.to_dict(orient="records") if not result.trades.empty else [],
            "equity_curve": {
                str(k): float(v) for k, v in result.equity_curve.items()
            },
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path
