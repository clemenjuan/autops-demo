#!/usr/bin/env python3
"""
Test script for the Iterative Reasoning Engine with Tool Loading
Tests multi-cycle thinking, planning, execution, and reflection with dynamic tool discovery
"""

import asyncio
import json
from agent.reasoning_engine import IterativeReasoningEngine
from agent.llm_interface import LLMInterface
from tools.tool_loader import load_tools

async def test_reasoning_with_tool_discovery():
    """Test the reasoning engine with dynamic tool discovery"""
    
    print("Testing Iterative Reasoning Engine with Dynamic Tool Discovery")
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
    
    # Create reasoning engine
    engine = IterativeReasoningEngine(
        reasoning_llm=reasoning_llm,
        general_llm=general_llm,
        tools=tools,
        tools_metadata=tools_metadata,
        max_iterations=3
    )
    
    # Test scenario: Earth observation query
    test_scenario = {
        'task_description': 'How many ships are currently in the Taiwan Strait?',
        'mission_context': {
            'mission_type': 'earth_observation',
            'priority': 'high'
        }
    }
    
    print(f"Test Query: {test_scenario['task_description']}")
    print()
    
    try:
        # Run iterative reasoning
        print("Starting reasoning process...")
        print("-" * 70)
        result = await engine.reason(test_scenario)
        
        print("\n✅ Reasoning Complete!")
        print("=" * 70)
        
        # Display results
        print(f"Situation Summary: {result.get('situation_summary', 'N/A')}")
        print(f"Confidence: {result.get('confidence', 0):.2f}")
        print(f"Total Reasoning Steps: {result.get('total_steps', 0)}")
        print(f"Task Status: {result.get('task_status', 'unknown')}")
        print()
        
        print("Analysis:")
        print(result.get('analysis', 'N/A'))
        print()
        
        print("Recommendations:")
        for i, rec in enumerate(result.get('recommendations', []), 1):
            print(f"  {i}. {rec}")
        print()
        
        # Show tool discovery
        if 'reasoning_trace' in result and len(result['reasoning_trace']) > 0:
            print("Tool Discovery & Reasoning Trace:")
            print("-" * 70)
            
            for i, step in enumerate(result['reasoning_trace'], 1):
                state = step.get('state', 'unknown')
                print(f"\n{i}. {state.upper()}")
                print(f"   Confidence: {step.get('confidence', 0):.2f}")
                print(f"   Reasoning: {step.get('reasoning', 'N/A')[:100]}...")
                
                if state == 'thinking' and i == 1:
                    # First iteration shows tool discovery
                    output = step.get('output_data', {})
                    if 'task_understanding' in output or 'relevant_tools' in output:
                        print(f"   🔧 Tools Discovered: {output.get('relevant_tools', 'N/A')}")
                
                if step.get('next_actions'):
                    print(f"   Next Actions: {step.get('next_actions', [])[:3]}")
        
        # Show tool results
        if result.get('tool_results'):
            print("\nTool Execution Results:")
            print("-" * 70)
            for tool_name, tool_result in result['tool_results'].items():
                print(f"{tool_name}:")
                print(f"  Status: {tool_result.get('status', 'unknown')}")
                print(f"  Message: {tool_result.get('message', 'N/A')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during reasoning: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_different_queries():
    """Test with different types of queries"""
    
    print("\n\nTesting Different Query Types")
    print("=" * 70)
    
    # Initialize
    general_llm = LLMInterface(preferred_model="auto", role="general")
    reasoning_llm = LLMInterface(preferred_model="auto", role="reasoning")
    tools, tools_metadata = load_tools()
    
    engine = IterativeReasoningEngine(
        reasoning_llm=reasoning_llm,
        general_llm=general_llm,
        tools=tools,
        tools_metadata=tools_metadata,
        max_iterations=2
    )
    
    # Different query types
    queries = [
        "Analyze satellite imagery of the Mediterranean Sea",
        "Detect vehicles in the border region",
        "Process optical imagery for ship detection",
        "Fuse SAR and optical data for comprehensive analysis"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 70)
        
        try:
            scenario = {'task_description': query}
            result = await engine.reason(scenario)
            
            print(f"✅ Status: {result.get('task_status', 'unknown')}")
            print(f"Confidence: {result.get('confidence', 0):.2f}")
            print(f"Steps: {result.get('total_steps', 0)}")
            
            # Show which tools were selected
            if result.get('reasoning_trace'):
                think_step = next((s for s in result['reasoning_trace'] if s.get('state') == 'thinking'), None)
                if think_step:
                    output = think_step.get('output_data', {})
                    relevant_tools = output.get('relevant_tools', [])
                    if relevant_tools:
                        print(f"Tools Selected: {', '.join(relevant_tools)}")
            
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
    print("Iterative Reasoning Engine Test Suite")
    print("Testing: Dynamic Tool Discovery & LLM-Driven Tool Selection")
    print("=" * 70)
    print()
    
    # Run tests
    asyncio.run(test_tool_loading())
    asyncio.run(test_reasoning_with_tool_discovery())
    asyncio.run(test_different_queries())
    
    print("\n" + "=" * 70)
    print("Testing Complete!")
