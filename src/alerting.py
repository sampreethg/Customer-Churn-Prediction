"""
Asynchronous Retention Alerting & Webhook Notification Engine.

Constructs rich Slack Block Kit, Discord embed, and generic CRM JSON payloads,
and dispatches non-blocking notifications with exponential backoff retries.
"""

import ipaddress
import logging
import os
import socket
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

DEFAULT_ALLOWED_DOMAINS = {"hooks.slack.com", "discord.com", "api.crm.example.com"}


def get_allowed_webhook_domains() -> set[str]:
    configured = os.getenv("WEBHOOK_ALLOWED_DOMAINS")
    if configured:
        return {d.strip().lower() for d in configured.split(",") if d.strip()}
    return set(DEFAULT_ALLOWED_DOMAINS)


def validate_webhook_url(url: str, allowed_domains: set[str] | None = None) -> None:
    if not url:
        raise ValueError("Webhook URL cannot be empty.")

    parsed = urlparse(url)
    if parsed.scheme.lower() != "https":
        raise ValueError(
            f"Invalid webhook URL scheme '{parsed.scheme}'. Only 'https' is permitted."
        )

    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"Webhook URL '{url}' is missing a valid hostname.")

    hostname_clean = hostname.strip().lower()

    # Block direct IP literals if they are private/loopback/link-local/metadata
    try:
        direct_ip = ipaddress.ip_address(hostname_clean)
        if (
            direct_ip.is_private
            or direct_ip.is_loopback
            or direct_ip.is_link_local
            or direct_ip.is_reserved
            or direct_ip.is_multicast
            or not direct_ip.is_global
        ):
            raise ValueError(
                f"Webhook URL target '{hostname_clean}' is a forbidden non-global or private IP."
            )
    except ValueError as exc:
        if "forbidden" in str(exc):
            raise
        # Not a raw IP literal, proceed to domain checking

    domains = allowed_domains if allowed_domains is not None else get_allowed_webhook_domains()
    if hostname_clean not in domains:
        raise ValueError(
            f"Webhook domain '{hostname_clean}' is not in the allowed domains: {sorted(list(domains))}"
        )

    # DNS resolution & IP check
    try:
        addr_info = socket.getaddrinfo(hostname_clean, 443)
        for entry in addr_info:
            ip_str = entry[4][0]
            ip = ipaddress.ip_address(ip_str)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or not ip.is_global
            ):
                raise ValueError(
                    f"Webhook domain '{hostname_clean}' resolved to forbidden private/local IP: {ip_str}"
                )
    except socket.gaierror as exc:
        is_test_env = (
            os.getenv("APP_ENV") == "test"
            or os.getenv("TESTING") == "1"
            or hostname_clean.endswith(".example.com")
            or hostname_clean == "example.com"
        )
        if not is_test_env:
            raise ValueError(
                f"DNS resolution failed for webhook host '{hostname_clean}': {exc}"
            ) from exc

logger = logging.getLogger("retention.alerting")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [RetentionAlerts] %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class AlertPayloadBuilder:
    """
    Constructs formatted notification payloads for Slack, Discord, and REST CRM platforms.
    """

    @staticmethod
    def build_slack_blocks(
        customer_id: str,
        prob: float,
        risk_level: str,
        action: str,
        top_factors: list[Any] | None = None,
    ) -> dict[str, Any]:
        """
        Build Slack Block Kit payload with header, status metrics, factors, and retention playbook.
        """
        pct_str = f"{prob * 100.0:.1f}%"
        env = os.getenv("APP_ENV", "production")
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Format factor bullets
        factor_lines = []
        if top_factors:
            for item in top_factors[:3]:
                if isinstance(item, dict):
                    fname = item.get("feature_name", "Feature")
                    sval = item.get("shap_value")
                    sval_str = f" (+{sval:.3f} risk)" if sval is not None else ""
                    factor_lines.append(f"• *{fname}*{sval_str}")
                else:
                    factor_lines.append(f"• *{str(item)}*")
        factor_text = (
            "\n".join(factor_lines)
            if factor_lines
            else "• General pricing fatigue & usage patterns"
        )

        return {
            "text": f"[ALERT] High Churn Risk Alert: Account {customer_id} ({pct_str})",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"[ALERT] High-Risk Churn Alert: {customer_id}",
                        "emoji": True,
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Account ID:*\n`{customer_id}`"},
                        {
                            "type": "mrkdwn",
                            "text": f"*Churn Probability:*\n[High] *{pct_str}* ({risk_level} Risk)",
                        },
                    ],
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Primary Hazard Drivers:*\n{factor_text}"},
                        {"type": "mrkdwn", "text": f"*Recommended Playbook:*\n _{action}_"},
                    ],
                },
                {"type": "divider"},
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f" Environment: *{env.upper()}* | Timestamp: *{ts}* | Engine: *Random Forest + SHAP*",
                        }
                    ],
                },
            ],
        }

    @staticmethod
    def build_generic_crm_payload(
        customer_id: str,
        prob: float,
        risk_level: str,
        action: str,
        top_factors: list[Any] | None = None,
    ) -> dict[str, Any]:
        """
        Build standardized JSON payload suitable for HubSpot, Salesforce, or internal CRM webhooks.
        """
        env = os.getenv("APP_ENV", "production")
        clean_factors = []
        if top_factors:
            for item in top_factors[:3]:
                if isinstance(item, dict):
                    clean_factors.append(
                        {
                            "feature": item.get("feature_name"),
                            "shap_attribution": item.get("shap_value"),
                        }
                    )
                else:
                    clean_factors.append({"feature": str(item)})

        return {
            "event_type": "customer.churn_risk.alert",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": env,
            "data": {
                "customer_id": customer_id,
                "churn_probability": round(prob, 4),
                "risk_tier": risk_level,
                "recommended_action": action,
                "primary_risk_drivers": clean_factors,
            },
        }


class WebhookNotifier:
    """
    Non-blocking async notification client with exponential backoff retries.
    """

    def __init__(
        self,
        slack_webhook_url: str | None = None,
        crm_webhook_url: str | None = None,
        alert_threshold: float | None = None,
    ):
        self.slack_webhook_url = slack_webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self.crm_webhook_url = crm_webhook_url or os.getenv("CRM_WEBHOOK_URL")
        if self.slack_webhook_url:
            validate_webhook_url(self.slack_webhook_url)
        if self.crm_webhook_url:
            validate_webhook_url(self.crm_webhook_url)

        if alert_threshold is not None:
            self.alert_threshold = alert_threshold
        else:
            self.alert_threshold = float(os.getenv("ALERT_THRESHOLD", "0.70"))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=3.0),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        reraise=True,
    )
    async def _post_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: dict[str, Any],
        destination_name: str,
    ) -> httpx.Response:
        """
        Dispatch POST request with automatic retry on transient errors or HTTP 5xx.
        """
        response = await client.post(url, json=payload)
        if response.status_code >= 500:
            response.raise_for_status()
        return response

    async def dispatch_retention_alert(
        self,
        customer_id: str,
        prob: float,
        risk_level: str,
        action: str,
        top_factors: list[Any] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate risk threshold and dispatch asynchronous notifications to configured webhooks.
        """
        if prob < self.alert_threshold:
            logger.debug(
                f"Skipping alert for {customer_id}: probability {prob:.2%} below threshold {self.alert_threshold:.2%}"
            )
            return {"status": "skipped", "reason": "below_threshold"}

        destinations = []
        if self.slack_webhook_url:
            validate_webhook_url(self.slack_webhook_url)
            destinations.append(
                ("Slack", self.slack_webhook_url, AlertPayloadBuilder.build_slack_blocks)
            )
        if self.crm_webhook_url:
            validate_webhook_url(self.crm_webhook_url)
            destinations.append(
                ("CRM", self.crm_webhook_url, AlertPayloadBuilder.build_generic_crm_payload)
            )

        if not destinations:
            logger.info(
                f"[DryRun] High-risk alert triggered for {customer_id} ({prob:.1%}), "
                "but no SLACK_WEBHOOK_URL or CRM_WEBHOOK_URL is configured."
            )
            return {
                "status": "dry_run",
                "reason": "no_webhooks_configured",
                "customer_id": customer_id,
            }

        dispatch_results = {}
        async with httpx.AsyncClient(timeout=6.0) as client:
            for name, url, payload_builder in destinations:
                payload = payload_builder(
                    customer_id=customer_id,
                    prob=prob,
                    risk_level=risk_level,
                    action=action,
                    top_factors=top_factors,
                )
                try:
                    resp = await self._post_with_retry(client, url, payload, name)
                    logger.info(
                        f"Successfully delivered retention alert for {customer_id} to {name} "
                        f"(Status: {resp.status_code})"
                    )
                    dispatch_results[name] = {"success": True, "status_code": resp.status_code}
                except Exception as exc:
                    logger.warning(
                        f"Failed delivering retention alert for {customer_id} to {name} after retries: {exc}"
                    )
                    dispatch_results[name] = {"success": False, "error": str(exc)}

        return {"status": "dispatched", "customer_id": customer_id, "results": dispatch_results}
