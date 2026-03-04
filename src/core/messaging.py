import logging
import select
import json
import time
from typing import Callable

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

logger = logging.getLogger("CORE.MESSAGING")

class PostgresNotifier:
    """
    Publishes events to a specific Postgres channel.
    Used by the ETL Hub to announce when new data is saved.
    """
    def __init__(self, db_url: str):
        self.db_url = db_url

    def notify(self, channel: str, payload: dict = None) -> None:
        """Broadcasts a message to all listeners on the channel."""
        try:
            # NOTIFY requires autocommit to be True
            conn = psycopg2.connect(self.db_url)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            
            with conn.cursor() as cur:
                if payload:
                    payload_str = json.dumps(payload)
                    # Use pg_notify for safe payload escaping
                    cur.execute("SELECT pg_notify(%s, %s);", (channel, payload_str))
                else:
                    cur.execute(f"NOTIFY {channel};")
                    
            conn.close()
            logger.debug(f"Successfully broadcasted to channel: '{channel}'")
            
        except Exception as e:
            logger.error(f"Failed to send NOTIFY on '{channel}': {e}")


class PostgresListener:
    """
    Listens for events on a specific Postgres channel.
    Used by the Trading Pods to wake up and evaluate strategies.
    """
    def __init__(self, db_url: str, channel: str, callback: Callable[[dict], None]):
        self.db_url = db_url
        self.channel = channel
        self.callback = callback
        self._running = False

    def start(self) -> None:
        """Starts the blocking event loop to listen for notifications."""
        self._running = True
        
        while self._running:
            try:
                conn = psycopg2.connect(self.db_url)
                conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                
                with conn.cursor() as cur:
                    cur.execute(f"LISTEN {self.channel};")
                
                logger.info(f"Listening for updates on channel '{self.channel}'...")
                
                while self._running:
                    # Wait for 5 seconds for activity on the connection.
                    # select.select puts the thread to sleep, using zero CPU.
                    if select.select([conn], [], [], 5) == ([], [], []):
                        continue 
                    
                    # If select returns, there is data. Poll the connection.
                    conn.poll()
                    
                    while conn.notifies:
                        notify = conn.notifies.pop(0)
                        payload = {}
                        if notify.payload:
                            try:
                                payload = json.loads(notify.payload)
                            except json.JSONDecodeError:
                                logger.warning(f"Received malformed JSON on '{self.channel}'")
                                
                        logger.info(f"Received Pub/Sub trigger on '{notify.channel}'!")
                        
                        # Trigger the Pod's execution pipeline
                        self.callback(payload)

            except psycopg2.OperationalError as e:
                logger.warning(f"Listener lost DB connection: {e}. Reconnecting in 5s...")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected Listener error on '{self.channel}': {e}")
                time.sleep(5)
            finally:
                if 'conn' in locals() and not conn.closed:
                    conn.close()

    def stop(self) -> None:
        """Safely terminates the listener loop."""
        self._running = False