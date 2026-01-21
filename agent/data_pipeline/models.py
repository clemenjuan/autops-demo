from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text, ForeignKey, Boolean, JSON
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


# ============================================================================
# Managed Satellites Models
# ============================================================================

class ManagedSatellite(Base):
    """Database record for managed satellites (links to TOON config)."""
    __tablename__ = 'managed_satellites'
    
    id = Column(Integer, primary_key=True)
    config_id = Column(String(50), unique=True, nullable=False, index=True)
    norad_id = Column(Integer, unique=True, index=True)
    name = Column(String(256), nullable=False)
    cospar_id = Column(String(20))
    description = Column(Text)
    
    # Current propulsion state (updated after maneuvers)
    fuel_remaining_kg = Column(Float)
    delta_v_remaining_m_s = Column(Float)
    
    # Operational status
    active = Column(Boolean, default=True)
    mission_start = Column(Date)
    mission_end_planned = Column(Date)
    
    last_state_update = Column(DateTime)
    last_maneuver = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'config_id': self.config_id,
            'norad_id': self.norad_id,
            'name': self.name,
            'cospar_id': self.cospar_id,
            'description': self.description,
            'fuel_remaining_kg': self.fuel_remaining_kg,
            'delta_v_remaining_m_s': self.delta_v_remaining_m_s,
            'active': self.active,
            'mission_start': self.mission_start.isoformat() if self.mission_start else None,
            'mission_end_planned': self.mission_end_planned.isoformat() if self.mission_end_planned else None,
            'last_state_update': self.last_state_update.isoformat() if self.last_state_update else None,
            'last_maneuver': self.last_maneuver.isoformat() if self.last_maneuver else None
        }


class StateVectorHistory(Base):
    """Precise state vectors from telemetry."""
    __tablename__ = 'state_vector_history'
    
    id = Column(Integer, primary_key=True)
    managed_satellite_id = Column(Integer, ForeignKey('managed_satellites.id'), nullable=False, index=True)
    epoch = Column(DateTime, nullable=False, index=True)
    
    # Position (ECI, meters)
    pos_x = Column(Float, nullable=False)
    pos_y = Column(Float, nullable=False)
    pos_z = Column(Float, nullable=False)
    
    # Velocity (ECI, m/s)
    vel_x = Column(Float, nullable=False)
    vel_y = Column(Float, nullable=False)
    vel_z = Column(Float, nullable=False)
    
    # Covariance matrix (6x6, stored as JSON array of 21 unique elements)
    covariance = Column(JSON)
    
    # Reference frame
    frame = Column(String(20), default='EME2000')
    
    # Source of this state vector
    source = Column(String(50), nullable=False)
    source_file = Column(String(256))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    managed_satellite = relationship("ManagedSatellite", backref="state_vectors")
    
    def to_dict(self):
        return {
            'id': self.id,
            'managed_satellite_id': self.managed_satellite_id,
            'epoch': self.epoch.isoformat() if self.epoch else None,
            'position_m': {'x': self.pos_x, 'y': self.pos_y, 'z': self.pos_z},
            'velocity_m_s': {'x': self.vel_x, 'y': self.vel_y, 'z': self.vel_z},
            'covariance': self.covariance,
            'frame': self.frame,
            'source': self.source
        }


class TelemetryPoint(Base):
    """Raw telemetry measurements."""
    __tablename__ = 'telemetry_points'
    
    id = Column(Integer, primary_key=True)
    managed_satellite_id = Column(Integer, ForeignKey('managed_satellites.id'), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Measurement type
    measurement_type = Column(String(50), nullable=False)
    
    # Ground station
    ground_station = Column(String(50))
    
    # Measurement data (flexible JSON)
    data = Column(JSON, nullable=False)
    
    # Quality indicator
    quality = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    managed_satellite = relationship("ManagedSatellite", backref="telemetry")


class ExecutedManeuver(Base):
    """Record of actually executed maneuvers."""
    __tablename__ = 'executed_maneuvers'
    
    id = Column(Integer, primary_key=True)
    managed_satellite_id = Column(Integer, ForeignKey('managed_satellites.id'), nullable=False, index=True)
    
    # Timing
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime)
    duration_s = Column(Float)
    
    # Maneuver type
    maneuver_type = Column(String(50))
    purpose = Column(String(256))
    
    # Commanded delta-v (m/s, in spacecraft frame or RTN)
    commanded_dv_x = Column(Float)
    commanded_dv_y = Column(Float)
    commanded_dv_z = Column(Float)
    commanded_dv_magnitude = Column(Float)
    
    # Achieved delta-v (from OD)
    achieved_dv_x = Column(Float)
    achieved_dv_y = Column(Float)
    achieved_dv_z = Column(Float)
    achieved_dv_magnitude = Column(Float)
    
    # Propulsion
    thrust_n = Column(Float)
    fuel_consumed_kg = Column(Float)
    
    # Pre/post orbit
    pre_sma_km = Column(Float)
    post_sma_km = Column(Float)
    pre_ecc = Column(Float)
    post_ecc = Column(Float)
    pre_inc_deg = Column(Float)
    post_inc_deg = Column(Float)
    
    # Status
    status = Column(String(20), default='completed')
    notes = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    managed_satellite = relationship("ManagedSatellite", backref="executed_maneuvers")
    
    def to_dict(self):
        return {
            'id': self.id,
            'managed_satellite_id': self.managed_satellite_id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'duration_s': self.duration_s,
            'maneuver_type': self.maneuver_type,
            'purpose': self.purpose,
            'commanded_dv_m_s': self.commanded_dv_magnitude,
            'achieved_dv_m_s': self.achieved_dv_magnitude,
            'fuel_consumed_kg': self.fuel_consumed_kg,
            'status': self.status
        }
