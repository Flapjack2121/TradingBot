"""Email + Telegram notifier for actionable signals.

Both channels are optional; the notifier silently skips a channel if its
credentials are missing.
"""
from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Iterable

import requests

log = logging.getLogger(__name__)


def format_signal_message(signal) -> str:
    reasons = "\n".join(
        f"  • {k}: {'✓' if v else '✗'}" for k, v in (signal.reasons or {}).items()
    )
    return (
        f"[{signal.strategy}] {signal.side} {signal.ticker}\n"
        f"  Entry      : {signal.entry}\n"
        f"  Stop Loss  : {signal.stop_loss}\n"
        f"  Take Profit: {signal.take_profit}\n"
        f"  Units      : {signal.units}\n"
        f"  Risk €     : {signal.risk_amount}\n"
        f"  Confidence : {signal.confidence:.0%}\n"
        f"{reasons}"
    )


@dataclass
class EmailConfig:
    smtp_host: str = ""
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    from_addr: str = ""
    to_addrs: list[str] = field(default_factory=list)

    @property
    def configured(self) -> bool:
        return bool(self.smtp_host and self.from_addr and self.to_addrs)


@dataclass
class TelegramConfig:
    bot_token: str = ""
    chat_id: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)


@dataclass
class Notifier:
    channels: list[str] = field(default_factory=list)
    email: EmailConfig = field(default_factory=EmailConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)

    def send_signals(self, signals: Iterable, header: str = "Trading signals") -> None:
        items = list(signals)
        if not items:
            return
        body = header + "\n\n" + "\n\n".join(format_signal_message(s) for s in items)
        if "email" in self.channels and self.email.configured:
            try:
                self._send_email(header, body)
            except Exception as exc:
                log.warning("Email failed: %s", exc)
        if "telegram" in self.channels and self.telegram.configured:
            try:
                self._send_telegram(body)
            except Exception as exc:
                log.warning("Telegram failed: %s", exc)

    # ── channels ──────────────────────────────────────────────────────────
    def _send_email(self, subject: str, body: str) -> None:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.email.from_addr
        msg["To"] = ", ".join(self.email.to_addrs)
        msg.set_content(body)
        with smtplib.SMTP(self.email.smtp_host, self.email.smtp_port) as smtp:
            smtp.starttls()
            if self.email.username:
                smtp.login(self.email.username, self.email.password)
            smtp.send_message(msg)

    def _send_telegram(self, text: str) -> None:
        url = f"https://api.telegram.org/bot{self.telegram.bot_token}/sendMessage"
        # Telegram has a 4096-char limit per message; chunk if necessary.
        for chunk_start in range(0, len(text), 3500):
            chunk = text[chunk_start: chunk_start + 3500]
            r = requests.post(
                url,
                data={"chat_id": self.telegram.chat_id, "text": chunk},
                timeout=15,
            )
            r.raise_for_status()
