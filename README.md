# AUTOPS: Autonomous Satellite Operations Assistant

AI-powered satellite operations system with advanced reasoning engine, collision risk assessment, Earth observation analysis, and autonomous constellation tasking.

## Quick Start

### Prerequisites
- Python 3.11+
- [UV](https://github.com/astral-sh/uv) package manager
- Docker (optional)

### Install UV
```bash
# Linux/macOS/WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Setup & Run

```bash
# Install dependencies
uv sync

# Set environment variables
export OLLAMA_HOST="https://ollama.sps.ed.tum.de"  # SPS local server, must be connected to TUM network
export OPENAI_API_KEY="your_key"  # OpenAI API

# WSL users: Suppress hardlink warning (optional)
export UV_LINK_MODE=copy

# Run the application
uv run python app.py
```

Access the dashboard at: http://localhost:5000

## Docker Deployment

```bash
# Build
docker build -t autops .

# Run
docker run -p 5000:5000 \
  -e OLLAMA_HOST="https://ollama.sps.ed.tum.de" \
  -e OPENAI_API_KEY="your_key" \
  autops
```

## Project Structure

```
autops-demo/
├── app.py                      # Flask API & main entry point
├── pyproject.toml              # UV dependencies & config
├── dockerfile                  # Docker deployment
├── agent/
│   ├── llm_interface.py        # Dual LLM interface (general + reasoning)
│   └── reasoning_engine.py     # Multi-cycle AI reasoning with tool discovery
├── tools/
│   ├── tools_metadata.json     # Tool definitions (metadata-driven)
│   ├── tool_loader.py          # Dynamic tool loading
│   ├── region_mapper_tool.py   # Geocoding with geopy
│   ├── image_processing_tool.py # Image processing (mock)
│   ├── object_detection_tool.py # Computer vision detection (mock)
│   └── data_fusion_tool.py     # Multi-source data fusion (mock)
├── templates/
│   └── index.html              # Web dashboard
├── static/
│   └── images/                 # Logos and icons
├── tests/
│   ├── test_llm_integration.py
│   ├── test_reasoning_engine.py
│   └── test_flask_api.py
└── simulation/
    └── orekit_simulator.py     # Orbital mechanics (future)
```

## Features

### AI-Driven Tool Orchestration
- **Natural Language Queries**: Submit tasks in plain English (e.g., "How many ships in Taiwan Strait?")
- **Dynamic Tool Discovery**: LLM analyzes task and discovers relevant tools automatically
- **Dual LLM Architecture**: 
  - `llama3.1:8b` for quick tasks (categorization, tool discovery)
  - `deepseek-r1:70b` for complex reasoning (planning, reflection)
- **Metadata-Driven Tools**: Tools defined in JSON, loaded dynamically
- **Iterative Reasoning**: Multi-cycle Think→Plan→Execute→Reflect loop
- **Validation Guardrails**: LLM-powered parameter validation with user fallback
- **Earth Observation Tools**:
  - Region mapping: async geocoding with geopy
  - Image processing, object detection, data fusion (mocks)

## Architecture

### Metadata-Driven Tool System
The system dynamically loads tools from JSON metadata and uses LLM reasoning to select and orchestrate them:

1. Tools defined in `tools_metadata.json`
2. Loaded dynamically by `tool_loader.py`
3. LLM discovers relevant tools during THINK phase
4. Reasoning engine plans and executes tool calls
5. Results synthesized in REFLECT phase

### Dual LLM Strategy
- **General LLM** (`llama3.1:8b`): Task categorization, tool discovery, quick analysis
- **Reasoning LLM** (`deepseek-r1:70b`): Complex reasoning, planning, reflection
- **Automatic fallback**: Ollama (free) → OpenAI (if unavailable)

### Reasoning Loop
1. **THINK**: Comprehensive task analysis (tool discovery, risk assessment, constraints)
2. **PLAN**: Create execution plan using discovered tools
3. **EXECUTE**: Run tools dynamically from registry
4. **REFLECT**: Evaluate results, decide if more iterations needed

## API Endpoints

- `POST /api/autonomous-tasking` - Submit natural language query (main endpoint)
- `GET /api/status` - System status and LLM availability
- `GET /api/task-history` - Task execution history

## Testing

```bash
# Test reasoning engine with tool discovery
uv run python tests/test_reasoning_engine.py

# Test LLM integration (Ollama/OpenAI)
uv run python tests/test_llm_integration.py

# Test Flask API endpoints (requires server running)
uv run python tests/test_flask_api.py
```

## Development

### Adding Dependencies
```bash
uv add package-name
```

### Project Configuration
All dependencies and settings are in `pyproject.toml`.

## Notes

- **Region Mapper**: Production-ready with async geopy, rate limiting, validation guardrails
- **Other Tools**: Image processing, object detection, data fusion are stubs
- **LLM Models**: `llama3.1:8b` (general), `deepseek-r1:70b` (reasoning)
- **Adding Tools**: Create `.py` file + JSON entry in metadata

## Contact

TUM Chair of Spacecraft Systems  
Clemente J. Juan Oliver  
clemente.juan@tum.de

---

**Supported by**: AUTOPS project, Bavarian Joint Research Program (BayVFP), MRF-2307-0004
