"""Thin data-access layer for the trade journal."""
from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from .models import Base, SignalLog, Trade, TradeStatus


class TradeRepo:
    def __init__(self, db_path: str | Path = "database/trades.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}", future=True)
        Base.metadata.create_all(self.engine)
        self._Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self._Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # ── trades ────────────────────────────────────────────────────────────
    def add_trade(self, **kwargs) -> Trade:
        with self.session() as s:
            t = Trade(**kwargs)
            s.add(t)
            s.flush()
            s.refresh(t)
            return t

    def update_status(self, trade_id: int, status: TradeStatus, **fields) -> Trade | None:
        with self.session() as s:
            t = s.get(Trade, trade_id)
            if t is None:
                return None
            t.status = status
            for k, v in fields.items():
                setattr(t, k, v)
            t.updated_at = dt.datetime.utcnow()
            s.flush()
            s.refresh(t)
            return t

    def close_trade(self, trade_id: int, exit_price: float, when: dt.datetime | None = None) -> Trade | None:
        with self.session() as s:
            t = s.get(Trade, trade_id)
            if t is None or t.status != TradeStatus.OPEN:
                return None
            t.exit_price = exit_price
            t.exit_at = when or dt.datetime.utcnow()
            pnl = (exit_price - t.entry) * t.units
            t.pnl = round(pnl, 2)
            t.pnl_pct = round((exit_price / t.entry - 1) * 100, 4) if t.entry else None
            risk_per_unit = (t.entry - t.stop_loss) if t.stop_loss else None
            if risk_per_unit and risk_per_unit > 0:
                t.realized_r = round((exit_price - t.entry) / risk_per_unit, 3)
            t.status = TradeStatus.CLOSED_WIN if pnl >= 0 else TradeStatus.CLOSED_LOSS
            t.updated_at = dt.datetime.utcnow()
            s.flush()
            s.refresh(t)
            return t

    def list_trades(self, status: TradeStatus | None = None) -> list[Trade]:
        with self.session() as s:
            stmt = select(Trade).order_by(Trade.created_at.desc())
            if status is not None:
                stmt = stmt.where(Trade.status == status)
            return list(s.scalars(stmt).all())

    def open_trades(self) -> list[Trade]:
        return self.list_trades(TradeStatus.OPEN)

    def to_dataframe(self) -> pd.DataFrame:
        with self.session() as s:
            rows = s.scalars(select(Trade).order_by(Trade.created_at.desc())).all()
            return pd.DataFrame([
                {
                    "id": t.id, "ticker": t.ticker, "strategy": t.strategy,
                    "status": t.status.value, "entry": t.entry,
                    "stop_loss": t.stop_loss, "take_profit": t.take_profit,
                    "units": t.units, "risk_€": t.risk_amount,
                    "position_€": t.position_value, "exit": t.exit_price,
                    "pnl_€": t.pnl, "pnl_%": t.pnl_pct, "realized_R": t.realized_r,
                    "created_at": t.created_at, "exit_at": t.exit_at,
                    "notes": t.notes,
                }
                for t in rows
            ])

    def stats(self) -> dict:
        df = self.to_dataframe()
        if df.empty:
            return {"trades": 0}
        closed = df[df["status"].isin(["closed_win", "closed_loss"])]
        wins = closed[closed["pnl_€"] > 0]
        losses = closed[closed["pnl_€"] <= 0]
        total_pnl = closed["pnl_€"].sum() if not closed.empty else 0
        return {
            "trades": len(df),
            "open": int((df["status"] == "open").sum()),
            "closed": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(closed) * 100, 2) if len(closed) else 0.0,
            "total_pnl": round(float(total_pnl), 2),
            "avg_R": round(float(closed["realized_R"].mean()), 3) if not closed.empty else None,
        }

    # ── signal log ────────────────────────────────────────────────────────
    def log_signals(self, signals: Iterable) -> int:
        count = 0
        with self.session() as s:
            for sig in signals:
                s.add(SignalLog(
                    ticker=sig.ticker,
                    strategy=sig.strategy,
                    side=sig.side,
                    confidence=sig.confidence,
                    entry=sig.entry,
                    stop_loss=sig.stop_loss,
                    take_profit=sig.take_profit,
                    payload=sig.to_dict(),
                    as_of=sig.as_of.to_pydatetime() if sig.as_of is not None else None,
                ))
                count += 1
        return count
