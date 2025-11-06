# PhD Research Progress Tracker

## Development Guidelines
- **Professional coding**: Keep it short, clean, simple. No unnecessary files. No emojis.
- **Preserve functionality**: Comment out unused code, don't delete. Keep for future development.
- **Documentation**: Always keep README.md updated and code well-explained.
- **Project organization**: Tests in `tests/` folder, templates in `templates/` (Flask convention), model files gitignored.

### Data Validation (Pydantic)
- **Goal**: Validate incoming API payloads and tool outputs.
- **Approach**: Use minimal Pydantic models for strict types and defaults.
- **Rules**:
  - Required fields must be explicit.
  - Constrain values (enums, ranges) where possible.
  - Provide safe defaults for optional fields.
  
## Project: Multi-modal Agentic Autonomous Systems in Satellite Operations

### Current Phase: Demo Development
**Goal**: Create a working demo with local LLM integration and basic tool-calling orchestration

### Completed Tasks (Major Milestones)
- [x] Project structure and dependency management (UV)
- [x] Dual LLM integration (Ollama → OpenAI fallback)
- [x] Metadata-driven tool system (JSON-based definitions)
- [x] Dynamic tool loading and discovery
- [x] Iterative reasoning engine (Think → Plan → Execute → Reflect)
- [x] LLM-driven tool discovery during THINK phase
- [x] Web interface with TUM branding
- [x] Real-time reasoning visualization
- [x] Natural language query interface
- [x] Confidence tracking and self-evaluation
- [x] Autonomous loop control
- [x] Region mapper tool (geopy, async, production-ready)
- [x] Validation guardrails (3-layer: metadata, planning, execution)
- [x] User input request system (generic, works for any tool)

### Current Status
**Architecture**: Metadata-driven tool orchestration with dual LLM reasoning

**What's Working**:
- ✅ Metadata-driven tool system (JSON → dynamic loading)
- ✅ Dual LLM architecture (llama3.1:8b + deepseek-r1:70b)
- ✅ LLM tool discovery in THINK phase
- ✅ Natural language query interface
- ✅ Region mapper: geopy geocoding, async, validation
- ✅ 3-layer validation: metadata → planning → execution
- ✅ User input request system (generic for all tools)
- ✅ 3 EO tool stubs (image_processor, object_detector, data_fusion)

**Next Steps**:
- [ ] Hierarchical reasoning with task stack (main task + subtasks)
- [ ] Long-term memory system to preserve context across task hierarchy
- [ ] Tool retry mechanism until correct output
- [ ] Implement actual EO tool logic (currently return mock data)
- [ ] Real satellite imagery integration
- [ ] Computer vision model integration
- [ ] Multi-agent coordination framework

### Development Roadmap

#### Phase 1: Foundation ✅ COMPLETED
- [x] Metadata-driven tool architecture
- [x] Dual LLM reasoning system
- [x] Dynamic tool discovery
- [x] Web interface with reasoning visualization
- [x] Natural language query interface

#### Phase 2: Tool Implementation (Current)
**2.1 Completed**:
- [x] Region mapper: geopy async geocoding with rate limiting
- [x] 3-layer validation guardrails (metadata, planning, execution)
- [x] Generic user input request system

**2.2 Next**:
- [ ] Image processing: satellite imagery preprocessing
- [ ] Object detection: computer vision integration (YOLOv5)
- [ ] Data fusion: multi-source synthesis
- [ ] Real satellite imagery database

**2.3 Future - Hierarchical Reasoning**:
- [ ] TaskContext, TaskStack, MemoryManager
- [ ] RetryController with tool validation
- [ ] Enhanced reasoning engine with subtask spawning
- [ ] Result propagation from subtasks to parent tasks

**2.2 Tool Implementation**:
- [ ] Implement EO tool logic (region mapping, image processing, detection, fusion)
- [ ] Real satellite imagery database integration
- [ ] Computer vision model integration (YOLOv5 or similar)
- [ ] Testing with real Earth observation scenarios

#### Phase 3: Advanced Reasoning
- [ ] Multi-agent coordination framework
- [ ] ReAct framework for decision-making
- [ ] Chain-of-Thought optimization
- [ ] LATS for constellation coordination
- [ ] RAG for domain knowledge

#### Phase 4: Production & Research
- [ ] Real-time data integration (CDMs, telemetry)
- [ ] Orekit simulation integration
- [ ] Performance benchmarking
- [ ] Academic paper preparation

### Research Questions
- How do multi-agent systems scale in space operations?
- What's the optimal human-AI interaction paradigm?
- How can RL agents learn from sparse, high-stakes data?
- What are the safety guarantees in autonomous decisions?

### Research Integration: Advanced AI Agent Reasoning Loops

**Research Context**: Integrate cutting-edge AI agent reasoning architectures into autonomous satellite operations, moving beyond traditional reactive systems to sophisticated multi-step reasoning loops. This research direction bridges existing work on multi-agent reinforcement learning (MARL) for satellite constellations with emerging advances in agentic AI systems.

**Key Research Direction: Cognitive Reasoning Architectures for Space Systems**

**Core Research Questions:**
- How can ReAct (Reasoning and Acting) frameworks be adapted for real-time satellite decision-making with orbital dynamics constraints?
- Can Language Agent Tree Search (LATS) architectures improve satellite constellation coordination through distributed reasoning?
- How do Chain-of-Thought reasoning patterns enhance satellite anomaly detection and autonomous response capabilities?

**Technical Implementation Framework:**

**Architecture Components:**
- **Perception Layer**: Telemetry processing, log message analysis, environmental sensing
- **Reasoning Engine**: Multi-step CoT reasoning for satellite state assessment and decision planning
- **Action Planning**: ReAct-style Thought-Action-Observation loops for satellite operations
- **Reflection Module**: Post-action evaluation and learning from mission outcomes
- **Multi-Agent Coordination**: LATS-inspired tree search for constellation-level task allocation

**Integration with Satellite Digital Twin (SDT):**
- Finite State Machine modeling of satellite operations as reasoning context
- Model-based simulation environment for testing agent reasoning algorithms
- Reinforcement Learning with Human Feedback (RLHF) integration for safety-critical operations
- Retrieval-Augmented Generation (RAG) for satellite domain knowledge integration

### Technical Architecture
- **Orchestrator Agent**: LLM-driven tool-calling system
- **Tool Registry**: Modular framework for collision avoidance, risk assessment, mission planning
- **Hybrid Reasoning**: LLM + algorithmic decision logic
- **Observability Layer**: Real-time data gathering and context tracking
- **Simulation Integration**: Orekit-based scenario testing
- **Operator Interface**: Natural language dialogue with explainable AI

### Key Innovations

**Metadata-Driven Tool System** (Nov 2025)
- JSON-based tool definitions, dynamic discovery
- LLM analyzes tasks and selects tools automatically
- No hardcoded workflows

**3-Layer Validation Guardrails** (Nov 2025)
- Layer 1: Enhanced metadata with parameter requirements
- Layer 2: Planning prompt includes tool specifications
- Layer 3: LLM-powered parameter validation + user fallback

**Dual LLM Architecture**
- `llama3.1:8b` for quick tasks, `deepseek-r1:70b` for reasoning
- Automatic Ollama → OpenAI fallback

**Iterative Reasoning Engine**
- Multi-cycle Think→Plan→Execute→Reflect loop
- Confidence tracking (0.0-1.0) at each phase
- Real-time visualization in UI

### Research Vision

**Goal**: Autonomous satellite operations with AI-driven reasoning and tool orchestration

**Inspired By**:
- NRO's vision for autonomous satellite management
- Big Impact CubeSat onboard autonomy
- Multi-agent reinforcement learning for constellations

**Current Approach**:
- Natural language queries → LLM reasoning → Dynamic tool selection
- Context-aware intelligence adapting to any task type
- No hardcoded workflows - pure metadata-driven orchestration

**Example**: "How many ships in the Taiwan Strait?"
1. LLM discovers relevant tools (region_mapper, object_detector, data_fusion)
2. Plans execution sequence
3. Executes tools dynamically
4. Synthesizes results with confidence scoring

### Current Architecture

**Core Components**:
1. **Reasoning Engine** (`agent/reasoning_engine.py`)
   - Think: Comprehensive task analysis (tool discovery, risk assessment, constraints)
   - Plan: Create execution plan with discovered tools
   - Execute: Run tools dynamically from registry
   - Reflect: Evaluate results + decide continuation

2. **Dual LLM Interface** (`agent/llm_interface.py`)
   - General LLM: llama3.1:8b for quick tasks
   - Reasoning LLM: deepseek-r1:70b for complex reasoning
   - Automatic fallback: Ollama → OpenAI

3. **Tool System** (`tools/`)
   - `tools_metadata.json`: Tool definitions (description, tags, parameters)
   - `tool_loader.py`: Dynamic import and instantiation
   - Individual tools: region_mapper, image_processor, object_detector, data_fusion

4. **Web Interface** (`templates/index.html`)
   - Natural language query submission
   - Real-time reasoning timeline visualization
   - TUM branding and professional design

### Hierarchical Reasoning Architecture (Planned)

**Goal**: Multi-level task execution with context preservation and automatic retry

**Architecture Design**:

**1. Task Hierarchy System**
   - **Task Stack**: LIFO structure for managing task hierarchy
   - **Task Context**: Each task preserves its own state and dependencies
   - **Task Types**: Main task, Subtask, Tool retry task
   
**2. Long-Term Memory**
   - **Main Task Context**: Preserved throughout entire execution
   - **Subtask Context**: Temporary context for subtask execution
   - **Cross-Task Memory**: Shared knowledge accessible by all tasks
   - **Result Propagation**: Subtask results bubble up to parent task

**3. Retry Mechanism**
   - **Tool Validation**: Check tool output against expected format
   - **Automatic Retry**: Retry failed tools with adjusted parameters
   - **Max Retry Limit**: Configurable retry count (default: 3)
   - **Fallback Strategy**: Alternative tools if primary fails
   
**4. Execution Flow**:
```
Main Task (preserved in memory)
  ├─ Think → Plan → Execute
  │   ├─ Tool fails → Create retry subtask
  │   │   └─ Retry with adjusted params (1..N attempts)
  │   ├─ Complex action → Create subtask
  │   │   └─ Think → Plan → Execute (independent reasoning)
  │   └─ Return to main task with results
  └─ Reflect → Continue or Complete
```

**5. Memory Structure**:
```python
{
  "main_task": {
    "description": "Original user query",
    "context": "Full context and requirements",
    "status": "in_progress"
  },
  "task_stack": [
    {"type": "main", "context": {...}},
    {"type": "subtask", "parent_id": "main", "context": {...}}
  ],
  "shared_memory": {
    "discovered_tools": [...],
    "gathered_data": {...},
    "learned_constraints": [...]
  }
}
```

**6. Implementation Components**:
   - `TaskContext`: Dataclass for task state management
   - `TaskStack`: Stack manager for hierarchical execution
   - `MemoryManager`: Long-term context preservation
   - `RetryController`: Tool retry logic with validation
   - Enhanced `IterativeReasoningEngine`: Hierarchy-aware reasoning

**Benefits**:
   - Never lose track of main objective
   - Automatic recovery from tool failures
   - Decompose complex tasks into manageable subtasks
   - Maintain full execution history for debugging
   - Better handling of multi-step satellite operations

**API**:
- `POST /api/autonomous-tasking` - Submit NL query
- `GET /api/status` - System and LLM status
- `GET /api/task-history` - Execution history

**Adding New Tools**:
1. Create `toolname_tool.py` with `execute()` method
2. Add entry to `tools_metadata.json`
3. Tool automatically discovered and loaded

### Implementation Example: Hierarchical Reasoning

**Scenario**: "Count all ships in the Taiwan Strait and assess collision risk"

**Execution with Hierarchical System**:

```
MAIN TASK: Count ships and assess collision risk
├─ THINK: Analyze task → Need ship detection + risk assessment
├─ PLAN: 
│   1. Map Taiwan Strait region
│   2. Process satellite imagery
│   3. Detect ships
│   4. Assess collision risk
│
├─ EXECUTE Step 1: region_mapper
│   └─ SUCCESS → Store region bounds in shared memory
│
├─ EXECUTE Step 2: image_processor
│   └─ FAIL (corrupted image) 
│       ├─ CREATE SUBTASK: "Repair or find alternative imagery"
│       │   ├─ THINK: Check image quality issues
│       │   ├─ PLAN: Try alternative data source
│       │   ├─ EXECUTE: image_processor (retry with params)
│       │   └─ REFLECT: Success → Return to main task
│       └─ RESULT: Processed imagery available
│
├─ EXECUTE Step 3: object_detector
│   └─ PARTIAL SUCCESS (low confidence detections)
│       ├─ CREATE RETRY: Adjust detection threshold
│       │   ├─ Retry 1: threshold=0.4 → Better results
│       │   └─ VALIDATE: Confidence > 0.7 → Accept
│       └─ RESULT: 47 ships detected
│
├─ EXECUTE Step 4: collision_risk_tool
│   └─ BLOCKED (tool not available)
│       ├─ CREATE SUBTASK: "Manual risk assessment"
│       │   ├─ THINK: Calculate based on ship density
│       │   ├─ PLAN: Use ship positions + spacing
│       │   ├─ EXECUTE: Basic risk calculation
│       │   └─ REFLECT: Medium risk identified
│       └─ RESULT: Risk assessment complete
│
└─ REFLECT: All objectives met
    ├─ Ships counted: 47
    ├─ Risk level: Medium
    └─ Confidence: 0.75
```

**Memory State During Execution**:

```python
{
  "main_task": {
    "id": "task_001",
    "description": "Count ships and assess collision risk",
    "status": "in_progress",
    "created_at": "2025-11-05T12:00:00Z"
  },
  "task_stack": [
    {"id": "task_001", "type": "main", "depth": 0},
    {"id": "task_001_subtask_1", "type": "subtask", "depth": 1, "parent": "task_001"}
  ],
  "shared_memory": {
    "region_bounds": {"lat": [23.5, 26.5], "lon": [118, 122]},
    "processed_imagery": "data/taiwan_strait_20251105.tif",
    "detected_ships": 47,
    "ship_positions": [...],
    "risk_level": "medium"
  },
  "execution_history": [
    {"task": "task_001", "phase": "think", "timestamp": "..."},
    {"task": "task_001", "phase": "plan", "timestamp": "..."},
    {"task": "task_001", "phase": "execute", "tool": "region_mapper", "result": "success"},
    {"task": "task_001_subtask_1", "phase": "think", "timestamp": "..."},
    ...
  ]
}
```

**Key Features**:
- Main task context never lost during subtask execution
- Automatic retry when tools fail or return poor results
- Subtasks can spawn their own reasoning cycles
- Shared memory accessible across all task levels
- Full execution trace for debugging and analysis

### Notes
- Keep all tools and ideas (comment out unused ones)
- Focus on clean, maintainable and professional code (no emojis)
- Prioritize research contributions over feature completeness
- Document everything for reproducibility in an extremely concise, minimalist style
- This is a serious research tool for satellite operations - maintain professional standards