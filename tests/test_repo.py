from pathlib import Path

import pytest

from database import TradeRepo, TradeStatus


@pytest.fixture
def repo(tmp_path: Path) -> TradeRepo:
    return TradeRepo(tmp_path / "trades.db")


def test_add_and_close_trade(repo: TradeRepo) -> None:
    t = repo.add_trade(
        ticker="AAPL", strategy="triple_confirmation", side="BUY",
        entry=100.0, stop_loss=95.0, take_profit=110.0,
        units=20, risk_amount=100, position_value=2000, r_multiple=2,
        status=TradeStatus.OPEN,
    )
    assert t.id == 1
    closed = repo.close_trade(t.id, exit_price=110.0)
    assert closed.status == TradeStatus.CLOSED_WIN
    assert closed.pnl == pytest.approx(20 * 10)
    assert closed.realized_r == pytest.approx(2.0)


def test_close_loss(repo: TradeRepo) -> None:
    t = repo.add_trade(
        ticker="AAPL", strategy="x", entry=100, stop_loss=90, units=10,
        risk_amount=100, position_value=1000, status=TradeStatus.OPEN,
    )
    closed = repo.close_trade(t.id, exit_price=90.0)
    assert closed.status == TradeStatus.CLOSED_LOSS
    assert closed.pnl == pytest.approx(-100.0)
    assert closed.realized_r == pytest.approx(-1.0)


def test_stats(repo: TradeRepo) -> None:
    a = repo.add_trade(ticker="A", strategy="x", entry=100, stop_loss=90, units=10,
                       risk_amount=100, position_value=1000, status=TradeStatus.OPEN)
    b = repo.add_trade(ticker="B", strategy="x", entry=100, stop_loss=90, units=10,
                       risk_amount=100, position_value=1000, status=TradeStatus.OPEN)
    repo.close_trade(a.id, exit_price=120)   # +200 profit
    repo.close_trade(b.id, exit_price=80)    # -200 loss
    stats = repo.stats()
    assert stats["trades"] == 2
    assert stats["closed"] == 2
    assert stats["win_rate"] == 50.0
    assert stats["total_pnl"] == pytest.approx(0.0)
