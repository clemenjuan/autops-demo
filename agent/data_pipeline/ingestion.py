from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from apscheduler.schedulers.blocking import BlockingScheduler
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
    
    def sync_cycle(self):
        session = self.Session()
        fetch_start = datetime.utcnow()
        
        try:
            logger.info("Fetching satellite data from KeepTrack API")
            satellites = self.client.fetch_all()
            logger.info(f"Retrieved {len(satellites)} satellite records")
            
            # Normalize all satellites
            normalized_sats = []
            for raw_sat in satellites:
                normalized = self.client.normalize_satellite(raw_sat)
                if normalized['norad_id']:
                    normalized_sats.append(normalized)
            
            logger.info(f"Normalized {len(normalized_sats)} valid satellites")
            
            # Get existing NORAD IDs in one query
            existing_ids = {s.norad_id for s in session.query(Satellite.norad_id).all()}
            logger.info(f"Found {len(existing_ids)} existing satellites in database")
            
            # Bulk insert new satellites only
            new_sats = [s for s in normalized_sats if s['norad_id'] not in existing_ids]
            if new_sats:
                logger.info(f"Inserting {len(new_sats)} new satellites...")
                session.bulk_insert_mappings(Satellite, new_sats)
                session.flush()
            
            processed = len(normalized_sats)
            logger.info(f"Processed {processed} valid satellite records")
            
            # Build NORAD ID to satellite ID mapping
            logger.info("Building satellite ID mapping...")
            sat_id_map = {s.norad_id: s.id for s in session.query(Satellite.norad_id, Satellite.id).all()}
            
            # Prepare TLE records for bulk insert
            logger.info("Preparing TLE records...")
            tle_records = []
            for raw_sat in satellites:
                tle1 = raw_sat.get('tle1')
                tle2 = raw_sat.get('tle2')
                if not tle1 or not tle2:
                    continue
                
                norad_id = self.client.extract_norad_id(tle1)
                if not norad_id or norad_id not in sat_id_map:
                    continue
                
                orbital_params = self.client.parse_tle_orbital_params(tle1, tle2)
                
                tle_records.append({
                    'satellite_id': sat_id_map[norad_id],
                    'epoch': self.client.parse_tle_epoch(tle1),
                    'line1': tle1,
                    'line2': tle2,
                    'a': orbital_params['a'],
                    'e': orbital_params['e'],
                    'i': orbital_params['i'],
                    'raan': orbital_params['raan'],
                    'aop': orbital_params['aop'],
                    'mean_anomaly': orbital_params['mean_anomaly'],
                    'collected_at': fetch_start,
                    'source': 'keeptrack_v2'
                })
            
            if tle_records:
                logger.info(f"Inserting {len(tle_records)} TLE records...")
                session.bulk_insert_mappings(TLEHistory, tle_records)
                session.flush()
            
            tle_count = len(tle_records)
            
            maneuvers = self._detect_maneuvers(satellites, session)
            for m in maneuvers:
                session.add(m)
            session.flush()
            
            lineage = DataLineage(
                source='keeptrack_v2',
                fetch_timestamp=fetch_start,
                records_processed=processed,
                maneuvers_detected=len(maneuvers),
                response_hash=str(hash(str(len(satellites))))
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
            tle1 = sat.get('tle1')
            if not tle1:
                continue
            
            norad_id = self.client.extract_norad_id(tle1)
            if not norad_id:
                continue
                
            sat_record = session.query(Satellite).filter_by(norad_id=norad_id).first()
            if not sat_record:
                continue
            
            prev_tle = session.query(TLEHistory)\
                .filter(TLEHistory.satellite_id == sat_record.id)\
                .order_by(TLEHistory.collected_at.desc())\
                .offset(1)\
                .first()
            
            if not prev_tle:
                continue
        
        return maneuvers
    
    def start_scheduler(self):
        self.scheduler = BlockingScheduler()
        self.scheduler.add_job(
            self.sync_cycle,
            'cron',
            hour='*',
            minute=0,
            id='keeptrack_sync'
        )
        logger.info("Ingestion scheduler started (hourly sync)")
        self.scheduler.start()
    
    def stop_scheduler(self):
        if self.scheduler:
            self.scheduler.shutdown()
            logger.info("Ingestion scheduler stopped")
