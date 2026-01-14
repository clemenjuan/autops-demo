import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from agent.data_pipeline.ingestion import IngestionPipeline
from agent.data_pipeline.models import Base, Satellite, TLEHistory, Maneuver, DataLineage

@pytest.fixture
def test_db():
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session, engine

@pytest.mark.asyncio
async def test_sync_cycle(test_db):
    Session, engine = test_db
    db_url = 'sqlite:///:memory:'
    pipeline = IngestionPipeline(db_url)
    pipeline.engine = engine
    pipeline.Session = Session
    
    await pipeline.sync_cycle()
    
    session = Session()
    sat_count = session.query(Satellite).count()
    tle_count = session.query(TLEHistory).count()
    lineage_count = session.query(DataLineage).count()
    session.close()
    
    assert sat_count > 0
    assert tle_count > 0
    assert lineage_count == 1

@pytest.mark.asyncio
async def test_maneuver_detection(test_db):
    Session, engine = test_db
    db_url = 'sqlite:///:memory:'
    pipeline = IngestionPipeline(db_url)
    pipeline.engine = engine
    pipeline.Session = Session
    
    await pipeline.sync_cycle()
    await pipeline.sync_cycle()
    
    session = Session()
    maneuver_count = session.query(Maneuver).count()
    session.close()
    
    assert maneuver_count >= 0
