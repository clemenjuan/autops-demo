# AUTOPS: Autonomous Satellite Operations Assistant

AI-powered satellite operations system with CoALA reasoning engine, satellite data integration, and autonomous task orchestration.

## Quick Start

### Prerequisites
- Python 3.11+
- [UV](https://github.com/astral-sh/uv) package manager
- PostgreSQL 15+ (for satellite data pipeline)
- Docker (optional)

### Install UV
```bash
# Linux/macOS/WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Setup & Run

**Main Application (CoALA + Web UI):**
```bash
# Install dependencies
uv sync

# Set environment variables
export OLLAMA_HOST="https://ollama.sps.ed.tum.de"
export OPENAI_API_KEY="your_key"

# Run
uv run python app.py
```

Access the dashboard at: http://localhost:5000

**Satellite Data Pipeline (Optional):**
```bash
# Set up PostgreSQL
docker run --name autops-postgres \
  -e POSTGRES_PASSWORD=yourpassword \
  -e POSTGRES_DB=autops_db \
  -p 5432:5432 \
  -d postgres:15

# Apply schema
psql postgresql://postgres:yourpassword@localhost:5432/autops_db < migrations/001_init_schema.sql

# Configure
export DATABASE_URL="postgresql://postgres:yourpassword@localhost:5432/autops_db"

# Run satellite API server
python run_satellite_api.py  # Port 8000

# Run data ingestion (hourly sync from KeepTrack API)
python run_ingestion.py
```

## Project Structure

```
autops-demo/
├── app.py                      # Flask API & main entry point
├── pyproject.toml              # Dependencies & configuration
├── dockerfile                  # Docker deployment
├── agent/
│   ├── llm_interface.py        # Dual LLM interface
│   ├── coala_reasoning_engine.py # CoALA reasoning engine
│   ├── memory/                 # Memory modules (episodic, semantic, procedural, working)
│   ├── data_pipeline/          # Satellite data ingestion
│   │   ├── fetchers/keeptrack_client.py  # KeepTrack API client
│   │   ├── ingestion.py        # Hourly sync pipeline
│   │   ├── models.py           # SQLAlchemy ORM
│   │   └── config.py           # Database configuration
│   └── api/
│       └── main.py             # FastAPI satellite data REST API
├── tools/
│   ├── tools_metadata.json     # Tool definitions
│   ├── satellite_data_tool.py  # Satellite data retrieval tool
│   ├── region_mapper_tool.py   # Geocoding
│   └── ...                     # Other tools
├── migrations/
│   └── 001_init_schema.sql     # PostgreSQL schema
├── tests/
│   ├── test_keeptrack_client.py # API client tests
│   ├── test_ingestion.py       # Pipeline tests
│   ├── test_api.py             # REST API tests
│   └── ...                     # Other tests
├── run_satellite_api.py        # Start satellite data API
├── run_ingestion.py            # Start data ingestion scheduler
├── templates/index.html        # Web dashboard
└── data/memory/                # Memory persistence (TOON format)
```

## Features

### CoALA Reasoning Engine
- **Natural Language Queries**: Submit tasks in plain English
- **Dynamic Tool Discovery**: LLM analyzes and selects relevant tools automatically
- **Dual LLM Architecture**: 
  - `llama3.1:8b` for task preprocessing
  - `deepseek-r1:70b` for complex reasoning
- **Iterative Reasoning**: Multi-cycle Think→Plan→Execute→Reflect loop
- **Memory System**: Working, episodic, semantic, and procedural memory with TOON persistence

### Satellite Data Integration
- **37k+ Satellites**: Real-time data from KeepTrack API
- **TLE History**: Temporal orbital element tracking
- **Maneuver Detection**: Threshold-based orbital change detection
- **REST API**: FastAPI endpoints for satellite queries
- **CoALA Integration**: Satellite data tool for natural language queries
- **Hourly Sync**: Automated data ingestion pipeline

### Available Tools
- **satellite_data**: Query satellite metadata, TLE history, detected maneuvers
- **region_mapper**: Geocoding with geopy
- **image_processor**, **object_detector**, **data_fusion**: Placeholders for future development

## Architecture

### CoALA Reasoning Engine
1. **THINK**: Task analysis and tool discovery
2. **PLAN**: Create execution plan
3. **EXECUTE**: Run tools from registry
4. **REFLECT**: Evaluate results, iterate if needed

### Dual LLM Strategy
- **General LLM** (`llama3.1:8b`): Task preprocessing
- **Reasoning LLM** (`deepseek-r1:70b`): Complex reasoning and planning
- **Automatic fallback**: Ollama → OpenAI

### Satellite Data Pipeline
1. **KeepTrackClient**: Fetches 37k+ satellites from KeepTrack API v2
2. **IngestionPipeline**: Hourly sync with APScheduler
3. **PostgreSQL**: Stores satellites, TLE history, maneuvers, data lineage
4. **REST API**: FastAPI endpoints for querying data
5. **CoALA Tool**: Satellite data accessible via natural language

## API Endpoints

### Main Application (Flask, Port 5000)
- `POST /api/query` - Submit natural language query
- `GET /api/status` - System status and LLM availability
- `GET /api/task-history` - Task execution history
- `POST /api/memory/clear` - Clear memory modules
- `GET /api/memory/status` - Memory statistics

### Satellite Data API (FastAPI, Port 8000)
- `GET /satellites` - List satellites with filtering
- `GET /tle/{norad_id}/history` - TLE history for satellite
- `GET /maneuvers` - Detected orbital maneuvers
- `GET /status` - Data freshness and sync status

## Testing

```bash
# Install dependencies
uv sync

# Run unit tests
pytest tests/ -v

# Specific test suites
pytest tests/test_keeptrack_client.py -v  # KeepTrack API client
pytest tests/test_ingestion.py -v         # Data ingestion pipeline
pytest tests/test_api.py -v               # REST API endpoints
```

## TOON Format

Memory and tool metadata use **TOON format** (Token-Oriented Object Notation) for efficient LLM interactions, reducing token usage by 30-60% vs JSON.

```bash
# Convert JSON to TOON
uv run python utils/convert_metadata.py

# Clear memory
uv run python utils/clear_memory.py
```

## Development

```bash
# Add dependencies
uv add package-name

# Run tests
pytest tests/ -v

# Convert JSON → TOON
uv run python utils/convert_metadata.py
```

### Database Migrations
```bash
# Apply schema
psql $DATABASE_URL < migrations/001_init_schema.sql
```

## Contact

TUM Chair of Spacecraft Systems  
Clemente J. Juan Oliver  
clemente.juan@tum.de

---

**Supported by**: AUTOPS project, Bavarian Joint Research Program (BayVFP), MRF-2307-0004
