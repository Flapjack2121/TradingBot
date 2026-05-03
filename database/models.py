"""SQLAlchemy models for the trade journal."""
from __future__ import annotations

import datetime as dt
import enum

from sqlalchemy import (
    Column, DateTime, Enum, Float, ForeignKey, Integer, JSON, String, Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class TradeStatus(str, enum.Enum):
    PENDING = "pending"     # signal accepted, not yet filled
    OPEN = "open"           # filled, in market
    CLOSED_WIN = "closed_win"
    CLOSED_LOSS = "closed_loss"
    CANCELLED = "cancelled"


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(16), nullable=False, index=True)
    strategy = Column(String(64), nullable=False)
    side = Column(String(8), nullable=False, default="BUY")
    status = Column(Enum(TradeStatus), nullable=False, default=TradeStatus.PENDING, index=True)

    entry = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=True)
    units = Column(Float, nullable=False, default=0.0)
    risk_amount = Column(Float, nullable=False, default=0.0)
    position_value = Column(Float, nullable=False, default=0.0)
    r_multiple = Column(Float, nullable=True)

    exit_price = Column(Float, nullable=True)
    exit_at = Column(DateTime, nullable=True)
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    realized_r = Column(Float, nullable=True)

    notes = Column(Text, nullable=True)
    extras = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow, nullable=False)


class SignalLog(Base):
    """Append-only record of every signal evaluation."""
    __tablename__ = "signal_log"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(16), nullable=False, index=True)
    strategy = Column(String(64), nullable=False)
    side = Column(String(8), nullable=False)
    confidence = Column(Float, nullable=False, default=0.0)
    entry = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    payload = Column(JSON, nullable=True)
    as_of = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False, index=True)
