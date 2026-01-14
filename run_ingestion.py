import logging
from agent.data_pipeline.ingestion import IngestionPipeline
from agent.data_pipeline.config import DATABASE_URL

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    pipeline = IngestionPipeline(DATABASE_URL)
    
    logging.info("Running initial sync...")
    pipeline.sync_cycle()
    logging.info("Initial sync complete. Starting hourly scheduler...")
    
    try:
        pipeline.start_scheduler()
    except (KeyboardInterrupt, SystemExit):
        pipeline.stop_scheduler()
