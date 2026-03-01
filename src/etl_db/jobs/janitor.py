"""
Janitor service for cleaning up old data from the database.
Prunes records older than the configured retention period.
"""
import logging
from datetime import datetime, timedelta, timezone
from sqlalchemy import delete

from src.core.config import load_yaml_config
from src.etl_db.db import get_session, Candle1H, Candle1D

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Janitor:
    """Service for cleaning up old database records."""
    
    def __init__(self, config_path: str):
        """Initialize janitor with configuration."""
        self.config = load_yaml_config(config_path)
        logger.info("Janitor initialized")
    
    def cleanup_old_data(self):
        """Remove data older than the retention period from both tables."""
        retention_days = self.config.get("janitor", {}).get("retention_days", 180)
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
        
        session = get_session()
        
        try:
            # Clean up 1h candles
            result_1h = session.execute(
                delete(Candle1H).where(Candle1H.timestamp < cutoff_date)
            )
            deleted_1h = result_1h.rowcount
            
            # Clean up 1d candles
            result_1d = session.execute(
                delete(Candle1D).where(Candle1D.timestamp < cutoff_date)
            )
            deleted_1d = result_1d.rowcount
            
            session.commit()
            
            logger.info(
                f"Janitor cleanup completed: "
                f"Deleted {deleted_1h} 1h candles and {deleted_1d} 1d candles "
                f"older than {cutoff_date.date()}"
            )
            
            return deleted_1h + deleted_1d
        
        except Exception as e:
            session.rollback()
            logger.error(f"Error during cleanup: {e}", exc_info=True)
            raise
        finally:
            session.close()


def main():
    """Main entry point for janitor."""
    import sys
    
    # Default config path
    config_path = sys.argv[1] if len(sys.argv) > 1 else "src/etl_db/config.yaml"
    
    janitor = Janitor(config_path)
    janitor.cleanup_old_data()


if __name__ == "__main__":
    main()

