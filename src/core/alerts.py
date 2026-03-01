import requests
import logging

logger = logging.getLogger(__name__)

def send_discord_alert(webhook_url: str, title: str, message: str, color: int = 0x00ff00, user_id: str = "667127606788751383"):
    """
    Sends a formatted embed message to a Discord channel.
    Optionally tags a user by their Discord ID.
    Colors: Green (0x00ff00), Red (0xff0000), Blue (0x0000ff)
    """
    if not webhook_url:
        return

    payload = {
        "content": f"<@{user_id}>" if user_id else None,
        "embeds": [{
            "title": title,
            "description": message,
            "color": color
        }]
    }

    if payload["content"] is None:
        del payload["content"]

    try:
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to send Discord alert: {e}")
