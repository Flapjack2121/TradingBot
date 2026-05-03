from notifications import format_signal_message
from strategies.base import Signal


def test_format_signal_message_contains_fields() -> None:
    sig = Signal(
        ticker="AAPL", strategy="triple_confirmation", side="BUY",
        entry=100.0, stop_loss=95.0, take_profit=110.0, units=20.0,
        risk_amount=100.0, position_value=2000.0, confidence=0.66,
        reasons={"trend": True, "momentum": False},
    )
    msg = format_signal_message(sig)
    for needle in ("AAPL", "triple_confirmation", "BUY", "100", "95", "110",
                   "trend", "momentum"):
        assert needle in msg
