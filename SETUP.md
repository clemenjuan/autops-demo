# AUTOPS Setup Guide

Complete step-by-step guide to get AUTOPS running with satellite data visualization.

---

## Prerequisites

- **Docker** (for PostgreSQL)
- **Python 3.11+**
- **uv** (Python package manager)

---

## Step 1: Start PostgreSQL Database

**In PowerShell:**

```powershell
# Start PostgreSQL container
docker run -d `
  --name autops-postgres `
  -e POSTGRES_PASSWORD=autops123 `
  -e POSTGRES_DB=autops_db `
  -p 5432:5432 `
  postgres:15

# Verify it's running
docker ps
```

---

## Step 2: Apply Database Schema

```powershell
# Navigate to project directory
cd "C:\Users\Clemente\OneDrive - TUM\Research\Autonomous AI Agents\autops-demo"

# Apply schema (PowerShell syntax)
Get-Content migrations/001_init_schema.sql | docker exec -i autops-postgres psql -U postgres -d autops_db
```

**You should see:** `CREATE TABLE`, `CREATE INDEX` messages.

---

## Step 3: Install Python Dependencies

```powershell
# Sync dependencies
uv sync
```

---

## Step 4: Set Environment Variables

**Set this in EVERY new PowerShell terminal:**

```powershell
$env:DATABASE_URL="postgresql://postgres:autops123@localhost:5432/autops_db"
```

---

## Step 5: Run the Satellite API Server

**Terminal 1 (PowerShell):**

```powershell
cd "C:\Users\Clemente\OneDrive - TUM\Research\Autonomous AI Agents\autops-demo"
$env:DATABASE_URL="postgresql://postgres:autops123@localhost:5432/autops_db"
uv run python run_satellite_api.py
```

**Expected output:**
```
INFO:     Started server process [...]
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Keep this running!**

---

## Step 6: Run the Flask App

**Terminal 2 (PowerShell):**

```powershell
cd "C:\Users\Clemente\OneDrive - TUM\Research\Autonomous AI Agents\autops-demo"
uv run python app.py
```

**Expected output:**
```
 * Running on http://127.0.0.1:5000
```

**Keep this running!**

---

## Step 7: Ingest Satellite Data (First Time)

**Terminal 3 (PowerShell):**

```powershell
cd "C:\Users\Clemente\OneDrive - TUM\Research\Autonomous AI Agents\autops-demo"
$env:DATABASE_URL="postgresql://postgres:autops123@localhost:5432/autops_db"
uv run python run_ingestion.py
```

**Expected output:**
```
INFO:agent.data_pipeline.ingestion:Fetching satellite data from KeepTrack API
INFO:agent.data_pipeline.ingestion:Retrieved 34537 satellite records
INFO:agent.data_pipeline.ingestion:Normalized 34537 valid satellites
INFO:agent.data_pipeline.ingestion:Found 0 existing satellites in database
INFO:agent.data_pipeline.ingestion:Inserting 34537 new satellites...
INFO:agent.data_pipeline.ingestion:Building satellite ID mapping...
INFO:agent.data_pipeline.ingestion:Preparing TLE records...
INFO:agent.data_pipeline.ingestion:Inserting 34537 TLE records...
INFO:agent.data_pipeline.ingestion:Sync complete: 34537 satellites, 34537 TLE records, 0 maneuvers
```

**This takes 30-60 seconds.** The scheduler will then run hourly updates automatically.

**Press Ctrl+C** when you see the initial sync complete, or keep it running for hourly updates.

---

## Step 8: Access the Frontend

Open your browser to: **http://localhost:5000**

Click **"Satellite Tracker"** in the navigation.

Click **"Load Satellites"** to see all 34,537 satellites!

### Visualization Controls
- **Show Globe**: Opens the 3D Cesium.js visualization
- **Past Orbits**: Toggle solid lines showing last 45 minutes of trajectory
- **Predictions**: Toggle dashed lines showing next orbital period
- **All Orbits**: Toggle both past and prediction orbits
- **Labels**: Toggle satellite name labels
- **Focus**: Click "Focus" on any satellite to fly to its position

---

## Quick Reference: Daily Startup

When you restart your computer or close terminals:

### Terminal 1: Satellite API
```powershell
cd "C:\Users\Clemente\OneDrive - TUM\Research\Autonomous AI Agents\autops-demo"
$env:DATABASE_URL="postgresql://postgres:autops123@localhost:5432/autops_db"
uv run python run_satellite_api.py
```

### Terminal 2: Flask App
```powershell
cd "C:\Users\Clemente\OneDrive - TUM\Research\Autonomous AI Agents\autops-demo"
uv run python app.py
```

**That's it!** Database and satellite data persist between restarts.

---

## Optional: Manual Data Refresh

To manually update satellite data from KeepTrack API:

```powershell
cd "C:\Users\Clemente\OneDrive - TUM\Research\Autonomous AI Agents\autops-demo"
$env:DATABASE_URL="postgresql://postgres:autops123@localhost:5432/autops_db"
uv run python run_ingestion.py
# Press Ctrl+C after initial sync completes
```

---

## Troubleshooting

### Database connection failed
- Make sure Docker container is running: `docker ps`
- Restart container: `docker start autops-postgres`

### Port already in use (8000 or 5000)
```powershell
# Find process using port
netstat -ano | findstr :8000
netstat -ano | findstr :5000

# Kill process (replace PID)
taskkill /PID <PID> /F
```

### Frontend shows "Failed to fetch"
- Make sure Satellite API (Terminal 1) is running on port 8000
- Check: http://localhost:8000/status

### No satellites appear
- Run ingestion script (Step 7) to populate database
- Check API status: http://localhost:8000/status
- Should show `"status": "healthy"` with recent `last_sync` timestamp

---

## Optional: Install Orekit for High-Precision Propagation

Orekit provides high-fidelity orbital calculations. It's optional - the system works without it.

```powershell
# Install Orekit Python wrapper (requires Java JDK 11+)
uv pip install orekit

# Download Orekit data files (first run will auto-download)
# Data includes: ephemerides, Earth orientation parameters, leap seconds
```

**Note:** Orekit requires a Java JVM. If not installed, the `orekit_propagation` tool will gracefully return an error message and the frontend visualization will continue to work using satellite.js.

---

## Optional: TOON Format Validation

Tool metadata and memory files use [TOON format](https://toonformat.dev/). To convert JSON files or fix malformed TOON:

```powershell
uv run python utils/convert_metadata.py
```

This will:
- Convert any `.json` files to `.toon`
- Fix `.toon` files that contain JSON instead of TOON
- Leave valid TOON files unchanged

---

## Clean Slate (Reset Everything)

```powershell
# Stop and remove container
docker stop autops-postgres
docker rm autops-postgres

# Start fresh (repeat Steps 1-7)
```
