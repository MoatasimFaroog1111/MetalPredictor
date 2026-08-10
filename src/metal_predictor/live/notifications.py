from __future__ import annotations

import html
from typing import Iterable

import httpx2 as httpx

from metal_predictor.live.contracts import ForecastSnapshot


class TelegramForecastPublisher:
    """Minimal Telegram Bot API publisher with explicit allowlisted destinations."""

    def __init__(
        self,
        bot_token: str,
        chat_ids: Iterable[str],
        client: httpx.Client | None = None,
    ) -> None:
        token = bot_token.strip()
        chats = tuple(str(value).strip() for value in chat_ids if str(value).strip())
        if not token:
            raise ValueError("Telegram bot token is required.")
        if not chats:
            raise ValueError("At least one Telegram chat id is required.")
        self._token = token
        self._chat_ids = chats
        self._client = client or httpx.Client(timeout=20.0)

    def publish_forecast(self, snapshot: ForecastSnapshot) -> None:
        text = self.format_forecast(snapshot)
        for chat_id in self._chat_ids:
            self.send_text(chat_id, text)

    def send_text(self, chat_id: str, text: str) -> None:
        response = self._client.post(
            f"https://api.telegram.org/bot{self._token}/sendMessage",
            json={
                "chat_id": str(chat_id),
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("ok") is not True:
            raise RuntimeError(f"Telegram sendMessage failed: {payload}")

    @staticmethod
    def format_forecast(snapshot: ForecastSnapshot) -> str:
        baseline_arrow = "⬆️" if snapshot.baseline_direction == "UP" else "⬇️" if snapshot.baseline_direction == "DOWN" else "➡️"
        challenger_arrow = "⬆️" if snapshot.challenger_direction == "UP" else "⬇️" if snapshot.challenger_direction == "DOWN" else "➡️"
        baseline_pct = (pow(2.718281828459045, snapshot.baseline_log_return_1h) - 1.0) * 100.0
        challenger_pct = (pow(2.718281828459045, snapshot.challenger_log_return_1h) - 1.0) * 100.0
        source_note = (
            "متوافق مع مصدر التدريب"
            if snapshot.source_compatible_with_training
            else "مصدر حي مختلف عن HistData — للمراقبة البحثية"
        )
        return (
            "🥈 <b>Silver AI Forecast</b>\n\n"
            f"السعر الحالي: <b>${snapshot.current_price_usd_per_kg:,.2f}/kg</b>\n"
            f"الساعة: {html.escape(snapshot.feature_timestamp_utc.isoformat())}\n\n"
            f"Baseline {html.escape(snapshot.baseline_model)}: {baseline_arrow} "
            f"{snapshot.baseline_direction} ({baseline_pct:+.4f}%)\n"
            f"السعر المتوقع: ${snapshot.baseline_predicted_price_usd_per_kg:,.2f}/kg\n\n"
            f"Research {html.escape(snapshot.challenger_model)}: {challenger_arrow} "
            f"{snapshot.challenger_direction} ({challenger_pct:+.4f}%)\n"
            f"السعر المتوقع: ${snapshot.challenger_predicted_price_usd_per_kg:,.2f}/kg\n\n"
            f"جودة البيانات: {html.escape(snapshot.data_quality)}\n"
            f"المصدر: {html.escape(snapshot.source_provider)}\n"
            f"{html.escape(source_note)}\n\n"
            "⚠️ <b>Research only</b> — لا توجد إشارة BUY/SELL، والميزة التنبؤية لم تُثبت بعد."
        )
