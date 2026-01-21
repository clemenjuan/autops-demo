-- Migration: 002_managed_satellites
-- Description: Add tables for managed satellite telemetry and maneuver tracking

-- Managed Satellites table (links to TOON config)
CREATE TABLE IF NOT EXISTS managed_satellites (
    id SERIAL PRIMARY KEY,
    config_id VARCHAR(50) UNIQUE NOT NULL,
    norad_id INTEGER UNIQUE,
    name VARCHAR(256) NOT NULL,
    cospar_id VARCHAR(20),
    description TEXT,
    
    -- Current propulsion state
    fuel_remaining_kg FLOAT,
    delta_v_remaining_m_s FLOAT,
    
    -- Operational status
    active BOOLEAN DEFAULT TRUE,
    mission_start DATE,
    mission_end_planned DATE,
    
    last_state_update TIMESTAMP,
    last_maneuver TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_managed_satellites_config_id ON managed_satellites(config_id);
CREATE INDEX IF NOT EXISTS idx_managed_satellites_norad_id ON managed_satellites(norad_id);

-- State Vector History table
CREATE TABLE IF NOT EXISTS state_vector_history (
    id SERIAL PRIMARY KEY,
    managed_satellite_id INTEGER NOT NULL REFERENCES managed_satellites(id) ON DELETE CASCADE,
    epoch TIMESTAMP NOT NULL,
    
    -- Position (ECI, meters)
    pos_x FLOAT NOT NULL,
    pos_y FLOAT NOT NULL,
    pos_z FLOAT NOT NULL,
    
    -- Velocity (ECI, m/s)
    vel_x FLOAT NOT NULL,
    vel_y FLOAT NOT NULL,
    vel_z FLOAT NOT NULL,
    
    -- Covariance matrix (6x6, stored as JSON)
    covariance JSONB,
    
    -- Reference frame
    frame VARCHAR(20) DEFAULT 'EME2000',
    
    -- Source
    source VARCHAR(50) NOT NULL,
    source_file VARCHAR(256),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_state_vectors_satellite ON state_vector_history(managed_satellite_id);
CREATE INDEX IF NOT EXISTS idx_state_vectors_epoch ON state_vector_history(epoch);

-- Telemetry Points table
CREATE TABLE IF NOT EXISTS telemetry_points (
    id SERIAL PRIMARY KEY,
    managed_satellite_id INTEGER NOT NULL REFERENCES managed_satellites(id) ON DELETE CASCADE,
    timestamp TIMESTAMP NOT NULL,
    
    -- Measurement
    measurement_type VARCHAR(50) NOT NULL,
    ground_station VARCHAR(50),
    data JSONB NOT NULL,
    quality FLOAT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_telemetry_satellite ON telemetry_points(managed_satellite_id);
CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry_points(timestamp);

-- Executed Maneuvers table
CREATE TABLE IF NOT EXISTS executed_maneuvers (
    id SERIAL PRIMARY KEY,
    managed_satellite_id INTEGER NOT NULL REFERENCES managed_satellites(id) ON DELETE CASCADE,
    
    -- Timing
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    duration_s FLOAT,
    
    -- Type
    maneuver_type VARCHAR(50),
    purpose VARCHAR(256),
    
    -- Commanded delta-v (m/s)
    commanded_dv_x FLOAT,
    commanded_dv_y FLOAT,
    commanded_dv_z FLOAT,
    commanded_dv_magnitude FLOAT,
    
    -- Achieved delta-v
    achieved_dv_x FLOAT,
    achieved_dv_y FLOAT,
    achieved_dv_z FLOAT,
    achieved_dv_magnitude FLOAT,
    
    -- Propulsion
    thrust_n FLOAT,
    fuel_consumed_kg FLOAT,
    
    -- Pre/post orbit
    pre_sma_km FLOAT,
    post_sma_km FLOAT,
    pre_ecc FLOAT,
    post_ecc FLOAT,
    pre_inc_deg FLOAT,
    post_inc_deg FLOAT,
    
    -- Status
    status VARCHAR(20) DEFAULT 'completed',
    notes TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_executed_maneuvers_satellite ON executed_maneuvers(managed_satellite_id);
CREATE INDEX IF NOT EXISTS idx_executed_maneuvers_time ON executed_maneuvers(start_time);

-- Update trigger for managed_satellites
CREATE OR REPLACE FUNCTION update_managed_satellite_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_managed_satellite ON managed_satellites;
CREATE TRIGGER trigger_update_managed_satellite
    BEFORE UPDATE ON managed_satellites
    FOR EACH ROW
    EXECUTE FUNCTION update_managed_satellite_timestamp();
