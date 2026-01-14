from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Satellite(Base):
    __tablename__ = 'satellites'
    
    id = Column(Integer, primary_key=True)
    norad_id = Column(Integer, unique=True, nullable=False, index=True)
    keeptrack_id = Column(Integer)
    name = Column(String(256))
    country = Column(String(100))
    operator = Column(String(256))
    orbit_type = Column(String(20))
    mission_type = Column(String(256))
    payload = Column(String(256))
    launched = Column(Date)
    decay_date = Column(Date)
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'norad_id': self.norad_id,
            'name': self.name,
            'country': self.country,
            'operator': self.operator,
            'orbit_type': self.orbit_type,
            'mission_type': self.mission_type,
            'payload': self.payload,
            'launched': self.launched.isoformat() if self.launched else None,
            'decay_date': self.decay_date.isoformat() if self.decay_date else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }

class TLEHistory(Base):
    __tablename__ = 'tle_history'
    
    id = Column(Integer, primary_key=True)
    satellite_id = Column(Integer, ForeignKey('satellites.id'), nullable=False, index=True)
    epoch = Column(DateTime, nullable=False)
    line1 = Column(String(70), nullable=False)
    line2 = Column(String(70), nullable=False)
    a = Column(Float)
    e = Column(Float)
    i = Column(Float)
    raan = Column(Float)
    aop = Column(Float)
    mean_anomaly = Column(Float)
    collected_at = Column(DateTime, nullable=False, index=True)
    source = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    satellite = relationship("Satellite", backref="tle_history")

class Maneuver(Base):
    __tablename__ = 'maneuvers'
    
    id = Column(Integer, primary_key=True)
    satellite_id = Column(Integer, ForeignKey('satellites.id'), nullable=False, index=True)
    detection_date = Column(DateTime, nullable=False, index=True)
    delta_a = Column(Float)
    delta_e = Column(Float)
    delta_i = Column(Float)
    confidence = Column(Float, default=0.5)
    maneuver_type = Column(String(100))
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    satellite = relationship("Satellite", backref="maneuvers")

class DataLineage(Base):
    __tablename__ = 'data_lineage'
    
    id = Column(Integer, primary_key=True)
    source = Column(String(50), nullable=False)
    fetch_timestamp = Column(DateTime, nullable=False, index=True)
    records_processed = Column(Integer)
    maneuvers_detected = Column(Integer)
    response_hash = Column(String(256))
    error_log = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
