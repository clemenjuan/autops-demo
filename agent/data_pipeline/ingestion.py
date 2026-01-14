from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import asyncio
import logging
from typing import List, Dict

from agent.data_pipeline.fetchers.keeptrack_client import KeepTrackClient
from agent.data_pipeline.models import Satellite, TLEHistory, Maneuver, DataLineage

logger = logging.getLogger(__name__)

class IngestionPipeline:
    def __init__(self, db_url: str):
        self.client = KeepTrackClient()
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.scheduler = None
    
    async def sync_cycle(self):
        session = self.Session()
        fetch_start = datetime.utcnow()
        
        try:
            logger.info("Fetching satellite data from KeepTrack API")
            satellites = await self.client.fetch_all()
            logger.info(f"Retrieved {len(satellites)} satellite records")
            
            for raw_sat in satellites:
                normalized = self.client.normalize_satellite(raw_sat)
                existing = session.query(Satellite).filter_by(norad_id=normalized['norad_id']).first()
                if existing:
                    for key, value in normalized.items():
                        setattr(existing, key, value)
                else:
                    session.add(Satellite(**normalized))
            session.flush()
            
            tle_count = 0
            for raw_sat in satellites:
                sat_record = session.query(Satellite).filter_by(norad_id=raw_sat['satid']).first()
                if not sat_record:
                    continue
                
                tle_record = TLEHistory(
                    satellite_id=sat_record.id,
                    epoch=self.client.parse_tle_epoch(raw_sat['line1']),
                    line1=raw_sat['line1'],
                    line2=raw_sat['line2'],
                    a=raw_sat.get('semiMajorAxis'),
                    e=raw_sat.get('eccentricity'),
                    i=raw_sat.get('inclination'),
                    raan=raw_sat.get('raan'),
                    aop=raw_sat.get('argOfPerigee'),
                    mean_anomaly=raw_sat.get('meanAnomaly'),
                    collected_at=fetch_start,
                    source='keeptrack_v2'
                )
                session.add(tle_record)
                tle_count += 1
            session.flush()
            
            maneuvers = self._detect_maneuvers(satellites, session)
            for m in maneuvers:
                session.add(m)
            session.flush()
            
            lineage = DataLineage(
                source='keeptrack_v2',
                fetch_timestamp=fetch_start,
                records_processed=len(satellites),
                maneuvers_detected=len(maneuvers),
                response_hash=str(hash(str(sorted([s['satid'] for s in satellites]))))
            )
            session.add(lineage)
            
            session.commit()
            logger.info(f"Sync complete: {len(satellites)} satellites, {tle_count} TLE records, {len(maneuvers)} maneuvers")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Sync failed: {str(e)}", exc_info=True)
            raise
        finally:
            session.close()
    
    def _detect_maneuvers(self, current_sats: List[Dict], session) -> List[Maneuver]:
        maneuvers = []
        MANEUVER_THRESHOLD_A = 0.01
        MANEUVER_THRESHOLD_I = 0.005
        
        for sat in current_sats:
            sat_record = session.query(Satellite).filter_by(norad_id=sat['satid']).first()
            if not sat_record:
                continue
            
            prev_tle = session.query(TLEHistory)\
                .filter(TLEHistory.satellite_id == sat_record.id)\
                .order_by(TLEHistory.collected_at.desc())\
                .offset(1)\
                .first()
            
            if not prev_tle:
                continue
            
            curr_a = sat.get('semiMajorAxis') or 0
            curr_e = sat.get('eccentricity') or 0
            curr_i = sat.get('inclination') or 0
            
            delta_a = curr_a - (prev_tle.a or 0)
            delta_e = curr_e - (prev_tle.e or 0)
            delta_i = curr_i - (prev_tle.i or 0)
            
            if abs(delta_a) > MANEUVER_THRESHOLD_A or abs(delta_i) > MANEUVER_THRESHOLD_I:
                maneuver = Maneuver(
                    satellite_id=sat_record.id,
                    detection_date=datetime.utcnow(),
                    delta_a=delta_a,
                    delta_e=delta_e,
                    delta_i=delta_i,
                    confidence=0.5,
                    maneuver_type='unknown'
                )
                maneuvers.append(maneuver)
        
        return maneuvers
    
    def start_scheduler(self):
        self.scheduler = AsyncIOScheduler()
        self.scheduler.add_job(
            self.sync_cycle,
            'cron',
            hour='*',
            minute=0,
            id='keeptrack_sync'
        )
        self.scheduler.start()
        logger.info("Ingestion scheduler started (hourly sync)")
    
    def stop_scheduler(self):
        if self.scheduler:
            self.scheduler.shutdown()
            logger.info("Ingestion scheduler stopped")
