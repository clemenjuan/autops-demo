import asyncio
import logging
from agent.data_pipeline.ingestion import IngestionPipeline
from agent.data_pipeline.config import DATABASE_URL

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    pipeline = IngestionPipeline(DATABASE_URL)
    pipeline.start_scheduler()
    
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        pipeline.stop_scheduler()
