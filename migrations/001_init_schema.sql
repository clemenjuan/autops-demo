CREATE TABLE satellites (
    id SERIAL PRIMARY KEY,
    norad_id INTEGER UNIQUE NOT NULL,
    keeptrack_id INTEGER,
    name VARCHAR(256),
    country VARCHAR(100),
    operator VARCHAR(256),
    orbit_type VARCHAR(20),
    mission_type VARCHAR(256),
    payload VARCHAR(256),
    launched DATE,
    decay_date DATE,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE tle_history (
    id SERIAL PRIMARY KEY,
    satellite_id INTEGER NOT NULL REFERENCES satellites(id),
    epoch TIMESTAMP NOT NULL,
    line1 VARCHAR(70) NOT NULL,
    line2 VARCHAR(70) NOT NULL,
    a FLOAT,
    e FLOAT,
    i FLOAT,
    raan FLOAT,
    aop FLOAT,
    mean_anomaly FLOAT,
    collected_at TIMESTAMP NOT NULL,
    source VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_tle_satellite_time ON tle_history(satellite_id, collected_at DESC);
CREATE INDEX idx_tle_collected ON tle_history(collected_at DESC);

CREATE TABLE maneuvers (
    id SERIAL PRIMARY KEY,
    satellite_id INTEGER NOT NULL REFERENCES satellites(id),
    detection_date TIMESTAMP NOT NULL,
    delta_a FLOAT,
    delta_e FLOAT,
    delta_i FLOAT,
    confidence FLOAT DEFAULT 0.5,
    maneuver_type VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_maneuver_satellite ON maneuvers(satellite_id);
CREATE INDEX idx_maneuver_date ON maneuvers(detection_date DESC);

CREATE TABLE data_lineage (
    id SERIAL PRIMARY KEY,
    source VARCHAR(50) NOT NULL,
    fetch_timestamp TIMESTAMP NOT NULL,
    records_processed INTEGER,
    maneuvers_detected INTEGER,
    response_hash VARCHAR(256),
    error_log TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_lineage_timestamp ON data_lineage(fetch_timestamp DESC);
