#!/usr/bin/env python3
"""
Test script for the CoALA Reasoning Engine
Tests Planning ↔ Execution cycles with memory modules and action space
"""

import asyncio
import json
from agent.coala_reasoning_engine import CoALAReasoningEngine
from agent.memory import WorkingMemory, EpisodicMemory, SemanticMemory, ProceduralMemory
from agent.llm_interface import LLMInterface
from tools.tool_loader import load_tools

async def test_coala_reasoning():
    """Test the CoALA reasoning engine with memory modules"""
    
    print("Testing CoALA Reasoning Engine")
    print("=" * 70)
    
    # Initialize LLMs
    general_llm = LLMInterface(preferred_model="auto", role="general")
    reasoning_llm = LLMInterface(preferred_model="auto", role="reasoning")
    
    # Load tools from metadata
    tools, tools_metadata = load_tools()
    
    print(f"Loaded {len(tools)} tools from metadata:")
    for tool_name in tools.keys():
        print(f"  - {tool_name}")
    print()
    
    # Initialize memory modules
    working_memory = WorkingMemory(persistent=False)
    episodic_memory = EpisodicMemory()
    semantic_memory = SemanticMemory()
    procedural_memory = ProceduralMemory()
    
    print(f"Memory modules initialized:")
    print(f"  - Working memory: {working_memory.size()} entries")
    print(f"  - Episodic memory: {episodic_memory.size()} episodes")
    print(f"  - Semantic memory: {semantic_memory.size()} facts")
    print(f"  - Procedural memory: {procedural_memory.size()} procedures")
    print()
    
    # Create CoALA reasoning engine
    engine = CoALAReasoningEngine(
        reasoning_llm=reasoning_llm,
        general_llm=general_llm,
        tools=tools,
        tools_metadata=tools_metadata,
        working_memory=working_memory,
        episodic_memory=episodic_memory,
        semantic_memory=semantic_memory,
        procedural_memory=procedural_memory,
        max_cycles=3
    )
    
    # Test scenario: Earth observation query for Bayern
    test_scenario = {
        'task_description': 'Detect vehicles in München city center',
        'mission_context': {
            'mission_type': 'earth_observation',
            'priority': 'high',
            'region': 'Bayern'
        }
    }
    
    print(f"Test Query: {test_scenario['task_description']}")
    print()
    
    try:
        # Run CoALA reasoning
        print("Starting CoALA reasoning process...")
        print("-" * 70)
        result = await engine.reason(test_scenario)
        
        print("\n✅ CoALA Reasoning Complete!")
        print("=" * 70)
        
        # Display results
        print(f"Situation Summary: {result.get('situation_summary', 'N/A')}")
        print(f"Confidence: {result.get('confidence', 0):.2f}")
        print(f"Total Cycles: {result.get('total_cycles', 0)}")
        print(f"Task Status: {result.get('task_status', 'unknown')}")
        print()
        
        print("Analysis:")
        print(result.get('analysis', 'N/A'))
        print()
        
        print("Recommendations:")
        for i, rec in enumerate(result.get('recommendations', []), 1):
            print(f"  {i}. {rec}")
        print()
        
        # Show CoALA cycles
        if 'reasoning_trace' in result and len(result['reasoning_trace']) > 0:
            print("CoALA Cycle Trace (Planning ↔ Execution):")
            print("-" * 70)
            
            for i, step in enumerate(result['reasoning_trace'], 1):
                state = step.get('state', 'unknown')
                cycle = step.get('cycle', 0)
                action = step.get('action_selected', 'None')
                
                print(f"\n{i}. {state.upper()} - Cycle {cycle}")
                print(f"   Action: {action}")
                print(f"   Confidence: {step.get('confidence', 0):.2f}")
                print(f"   Reasoning: {step.get('reasoning', 'N/A')[:100]}...")
        
        # Show tool results
        if result.get('tool_results'):
            print("\nTool Execution Results:")
            print("-" * 70)
            for tool_name, tool_result in result['tool_results'].items():
                print(f"{tool_name}:")
                print(f"  Status: {tool_result.get('status', 'unknown')}")
                print(f"  Message: {tool_result.get('message', 'N/A')}")
        
        # Show memory statistics
        print("\nMemory Statistics After Task:")
        print("-" * 70)
        print(f"Episodic memory: {episodic_memory.size()} episodes")
        print(f"Semantic memory: {semantic_memory.size()} facts")
        print(f"Procedural memory: {procedural_memory.size()} procedures")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during reasoning: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_different_queries():
    """Test with different types of queries (Bayern focus)"""
    
    print("\n\nTesting Different Query Types (Bayern Focus)")
    print("=" * 70)
    
    # Initialize
    general_llm = LLMInterface(preferred_model="auto", role="general")
    reasoning_llm = LLMInterface(preferred_model="auto", role="reasoning")
    tools, tools_metadata = load_tools()
    
    # Initialize memory modules
    working_memory = WorkingMemory(persistent=False)
    episodic_memory = EpisodicMemory()
    semantic_memory = SemanticMemory()
    procedural_memory = ProceduralMemory()
    
    engine = CoALAReasoningEngine(
        reasoning_llm=reasoning_llm,
        general_llm=general_llm,
        tools=tools,
        tools_metadata=tools_metadata,
        working_memory=working_memory,
        episodic_memory=episodic_memory,
        semantic_memory=semantic_memory,
        procedural_memory=procedural_memory,
        max_cycles=2
    )
    
    # Different query types focused on Bayern
    queries = [
        "Analyze satellite imagery of Bayern",
        "Detect vehicles in München city center",
        "Process optical imagery of the Alps region",
        "Map the Isar river through Munich"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 70)
        
        try:
            scenario = {'task_description': query}
            result = await engine.reason(scenario)
            
            print(f"✅ Status: {result.get('task_status', 'unknown')}")
            print(f"Confidence: {result.get('confidence', 0):.2f}")
            print(f"Cycles: {result.get('total_cycles', 0)}")
            
            # Show which actions were executed
            actions = result.get('actions_executed', [])
            if actions:
                print(f"Actions Executed: {', '.join(actions)}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:100]}")

async def test_tool_loading():
    """Test that tools load correctly from JSON metadata"""
    
    print("\n\nTesting Tool Loading")
    print("=" * 70)
    
    try:
        tools, metadata = load_tools()
        
        print(f"✅ Successfully loaded {len(tools)} tools")
        print()
        
        for tool_name, tool_info in tools.items():
            print(f"Tool: {tool_name}")
            print(f"  Description: {tool_info['description']}")
            print(f"  Tags: {', '.join(tool_info['tags'])}")
            print(f"  Examples: {tool_info['examples'][0] if tool_info['examples'] else 'None'}")
            print()
        
        # Test tool execution
        print("Testing Tool Execution:")
        print("-" * 70)
        for tool_name, tool_info in tools.items():
            try:
                result = await tool_info['execute']({'test': 'param'})
                print(f"{tool_name}: {result.get('status', 'unknown')} - {result.get('message', 'N/A')}")
            except Exception as e:
                print(f"{tool_name}: ERROR - {e}")
        
    except Exception as e:
        print(f"❌ Error loading tools: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("CoALA Reasoning Engine Test Suite")
    print("Testing: Planning ↔ Execution Cycles with Memory Modules")
    print("=" * 70)
    print()
    
    # Run tests
    asyncio.run(test_tool_loading())
    asyncio.run(test_coala_reasoning())
    asyncio.run(test_different_queries())
    
    print("\n" + "=" * 70)
    print("CoALA Testing Complete!")
