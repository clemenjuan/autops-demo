from fastapi import FastAPI, Query
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
from typing import Optional

from agent.data_pipeline.models import Satellite, TLEHistory, Maneuver, DataLineage

app = FastAPI(title="AUTOPS Satellite Data API")
Session = None

def init_db(db_url: str):
    global Session
    from sqlalchemy import create_engine
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)

@app.get("/satellites")
async def list_satellites(
    constellation: Optional[str] = Query(None),
    operator: Optional[str] = Query(None),
    limit: int = Query(100, le=1000)
):
    session = Session()
    query = session.query(Satellite)
    
    if constellation:
        query = query.filter(Satellite.name.ilike(f"%{constellation}%"))
    if operator:
        query = query.filter(Satellite.operator == operator)
    
    total = query.count()
    results = query.limit(limit).all()
    session.close()
    
    return {
        'count': total,
        'data': [s.to_dict() for s in results]
    }

@app.get("/tle/{norad_id}/history")
async def tle_history(
    norad_id: int,
    days: int = Query(30, ge=1, le=365)
):
    session = Session()
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    sat = session.query(Satellite).filter_by(norad_id=norad_id).first()
    if not sat:
        session.close()
        return {'error': 'Satellite not found'}
    
    records = session.query(TLEHistory)\
        .filter(TLEHistory.satellite_id == sat.id)\
        .filter(TLEHistory.collected_at >= cutoff)\
        .order_by(TLEHistory.collected_at.desc())\
        .all()
    
    session.close()
    
    return {
        'satellite_id': norad_id,
        'satellite_name': sat.name,
        'record_count': len(records),
        'history': [
            {
                'epoch': r.epoch.isoformat(),
                'a': r.a,
                'e': r.e,
                'i': r.i,
                'raan': r.raan,
                'aop': r.aop,
                'mean_anomaly': r.mean_anomaly,
                'collected_at': r.collected_at.isoformat(),
                'source': r.source
            }
            for r in records
        ]
    }

@app.get("/maneuvers")
async def detected_maneuvers(
    satellite_id: Optional[int] = Query(None),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    days: int = Query(30, ge=1, le=365)
):
    session = Session()
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    query = session.query(Maneuver)\
        .filter(Maneuver.detection_date >= cutoff)\
        .filter(Maneuver.confidence >= min_confidence)
    
    if satellite_id:
        sat = session.query(Satellite).filter_by(norad_id=satellite_id).first()
        if sat:
            query = query.filter(Maneuver.satellite_id == sat.id)
    
    results = query.all()
    session.close()
    
    return {
        'count': len(results),
        'data': [
            {
                'satellite_id': m.satellite_id,
                'detection_date': m.detection_date.isoformat(),
                'delta_a': m.delta_a,
                'delta_e': m.delta_e,
                'delta_i': m.delta_i,
                'confidence': m.confidence,
                'maneuver_type': m.maneuver_type
            }
            for m in results
        ]
    }

@app.get("/status")
async def data_status():
    session = Session()
    
    latest = session.query(DataLineage)\
        .order_by(DataLineage.fetch_timestamp.desc())\
        .first()
    
    session.close()
    
    if not latest:
        return {'status': 'no_data', 'message': 'No synchronization records found'}
    
    freshness_minutes = (datetime.utcnow() - latest.fetch_timestamp).total_seconds() / 60
    
    return {
        'status': 'healthy' if freshness_minutes < 120 else 'stale',
        'last_sync': latest.fetch_timestamp.isoformat(),
        'freshness_minutes': round(freshness_minutes, 1),
        'records_processed': latest.records_processed,
        'maneuvers_detected': latest.maneuvers_detected
    }
