import requests
import logging
from datetime import datetime, timezone

# Standard colors for Discord embeds
COLORS = {
    "INFO": 0x3498db,     # Blue
    "WARNING": 0xf1c40f,  # Yellow
    "ERROR": 0xe74c3c,    # Red
    "CRITICAL": 0x9b59b6   # Purple
}

class DiscordHandler(logging.Handler):
    """
    A custom logging handler that sends records to a Discord Webhook.
    """
    def __init__(self, webhook_url: str, user_id: str = None, level=logging.ERROR):
        super().__init__(level)
        self.webhook_url = webhook_url
        self.user_id = user_id

    def emit(self, record):
        if not self.webhook_url:
            return

        try:
            # Format the log message using the attached formatter
            log_entry = self.format(record)
            level_name = record.levelname
            color = COLORS.get(level_name, 0x7f8c8d)

            # Mention the user only on ERROR or CRITICAL
            content = f" <@{self.user_id}>" if self.user_id and record.levelno >= logging.ERROR else ""

            payload = {
                "content": content,
                "embeds": [{
                    "title": f"System Alert: {level_name}",
                    "description": f"```\n{log_entry}\n```", # Added code blocks for readability
                    "color": color,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "footer": {"text": f"Logger: {record.name}"}
                }]
            }

            # Send to Discord
            response = requests.post(self.webhook_url, json=payload, timeout=5)
            response.raise_for_status()
            
        except Exception:
            # Prevent logging loops if Discord is down
            self.handleError(record)

def setup_discord_logging(webhook_url: str, user_id: str = None, level=logging.ERROR):
    """
    Attaches the DiscordHandler to the root logger.
    """
    if not webhook_url:
        return

    handler = DiscordHandler(webhook_url, user_id, level)
    
    # Standard format: Time - Name - Level - Message
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logging.getLogger().addHandler(handler)

def send_direct_discord_message(webhook_url: str, title: str, message: str, color: int = 0x00ff00):
    """
    Utility for non-logging manual alerts (e.g. startup/shutdown notifications).
    """
    if not webhook_url:
        return

    payload = {
        "embeds": [{
            "title": title,
            "description": message,
            "color": color,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }]
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=5)
        response.raise_for_status()
    except Exception as e:
        print(f"Manual Discord Alert Failed: {e}")