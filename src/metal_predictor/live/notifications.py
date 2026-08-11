from __future__ import annotations

import html
from typing import Iterable
from urllib.parse import urlparse

import httpx2 as httpx

from metal_predictor.live.contracts import ForecastSnapshot


class TelegramForecastPublisher:
    """Telegram Bot API publisher with allowlisted destinations and webhook setup."""

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
        payload = self._post("sendMessage", {
            "chat_id": str(chat_id),
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        })
        if payload.get("ok") is not True:
            raise RuntimeError("Telegram sendMessage was rejected by the Bot API.")

    def configure_webhook(self, public_base_url: str, secret_token: str) -> dict[str, object]:
        base = public_base_url.strip().rstrip("/")
        parsed = urlparse(base)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Telegram webhook requires a public HTTPS base URL.")
        secret = secret_token.strip()
        if not secret:
            raise ValueError("Telegram webhook secret token is required.")
        webhook_url = f"{base}/api/v1/telegram/webhook"
        payload = self._post("setWebhook", {
            "url": webhook_url,
            "secret_token": secret,
            "allowed_updates": ["message"],
            "drop_pending_updates": False,
        })
        if payload.get("ok") is not True:
            raise RuntimeError("Telegram setWebhook was rejected by the Bot API.")
        return {
            "configured": True,
            "url": webhook_url,
            "description": str(payload.get("description", "")),
        }

    def _post(self, method: str, payload: dict[str, object]) -> dict[str, object]:
        try:
            response = self._client.post(
                f"https://api.telegram.org/bot{self._token}/{method}",
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            raise RuntimeError(f"Telegram {method} transport request failed.") from None
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"Telegram {method} returned a non-object response.")
        return data

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
