# Testing Guide

## Quick Test (Tool Discovery Only)

**1. Restart the main app to load the new satellite_data tool:**
```bash
# Stop the running app (Ctrl+C)
# Restart it
uv run python app.py
```

**2. Test tool discovery:**
- Open http://localhost:5000
- Submit query: `"Get satellite data for NORAD ID 25544"`
- **Expected result**: CoALA should discover the `satellite_data` tool and try to use it
- **Expected error**: "Satellite API server not running" (this is good - it means the tool was discovered!)

If you see this error, the tool registration is working! ✅

---

## Full Integration Test (With Database)

To test the complete pipeline:

### Step 1: Set up PostgreSQL

```bash
# Start PostgreSQL in Docker
docker run --name autops-postgres \
  -e POSTGRES_PASSWORD=yourpassword \
  -e POSTGRES_DB=autops_db \
  -p 5432:5432 \
  -d postgres:15

# Apply schema
psql postgresql://postgres:yourpassword@localhost:5432/autops_db < migrations/001_init_schema.sql
```

### Step 2: Configure environment

```bash
# PowerShell
$env:DATABASE_URL="postgresql://postgres:yourpassword@localhost:5432/autops_db"

# Or create a .env file:
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/autops_db
```

### Step 3: Start satellite API server

```bash
# In a new terminal
python run_satellite_api.py
```

This will:
- Start FastAPI on port 8000
- Make API endpoints available

### Step 4: (Optional) Run data ingestion

```bash
# In another terminal
python run_ingestion.py
```

This will:
- Fetch 37k+ satellites from KeepTrack API
- Store in PostgreSQL
- Run hourly sync

**Note**: First sync takes ~5-10 minutes to fetch and store all satellites

### Step 5: Test via CoALA

- Restart main app if needed: `uv run python app.py`
- Open http://localhost:5000
- Try queries:
  - `"Get satellite data for NORAD ID 25544"` (ISS)
  - `"Show me TLE history for the ISS"`
  - `"Find recent maneuvers"`

---

## Troubleshooting

### Tool not discovered
- **Symptom**: "the necessary tools for retrieving satellite data are not available"
- **Fix**: Restart `app.py` to reload tools

### API connection refused
- **Symptom**: "Cannot connect to http://localhost:8000"
- **Fix**: Start the API server with `python run_satellite_api.py`

### Database errors
- **Symptom**: Database connection errors
- **Fix**: Check PostgreSQL is running and DATABASE_URL is set correctly

### Empty results
- **Symptom**: "Satellite not found" 
- **Fix**: Run `python run_ingestion.py` to populate database

---

## What to Test

### ✅ Minimal (Tool Discovery)
- Tool is registered and discoverable by CoALA
- CoALA attempts to use the tool
- Error messages are clear

### ✅ Integration (Full Pipeline)
- PostgreSQL schema creation
- Data ingestion from KeepTrack API
- REST API endpoints
- CoALA tool execution
- Data retrieval via natural language

### ⚠️ Not Yet Testable
- Frontend visualization (not in Phase 1 scope)
- 7-day continuous sync (requires runtime)
- Production deployment on TUM server
