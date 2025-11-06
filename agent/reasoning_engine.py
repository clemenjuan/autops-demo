import json
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

class ReasoningState(Enum):
    INITIAL = "initial"
    THINKING = "thinking"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    ERROR = "error"
    EXTERNAL = "external"

@dataclass
class ReasoningStep:
    state: ReasoningState
    timestamp: str
    input_data: Dict
    output_data: Dict
    confidence: float
    reasoning: str
    next_actions: List[str]
    message: str = ""
    
    def to_dict(self):
        base_dict = {
            'state': self.state.value,
            'timestamp': self.timestamp,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'next_actions': self.next_actions,
            'message': self.message
        }
        
        if self.output_data:
            if self.state == ReasoningState.THINKING:
                base_dict['analysis'] = self.output_data.get('problem_analysis', '')
                base_dict['discovered_tools'] = self.output_data.get('discovered_tools', [])
                base_dict['details'] = self.output_data
            elif self.state == ReasoningState.PLANNING:
                base_dict['plan'] = self.output_data.get('expected_outcomes', '')
                base_dict['actions'] = self.output_data.get('execution_plan', [])
            elif self.state == ReasoningState.EXECUTING:
                base_dict['tool_results'] = self.output_data.get('results', {})
                base_dict['execution_details'] = self.output_data.get('execution_summary', '')
                base_dict['actions_executed'] = self.output_data.get('actions_executed', [])
            elif self.state == ReasoningState.REFLECTING:
                base_dict['evaluation'] = self.output_data.get('evaluation', '')
                base_dict['should_continue'] = self.output_data.get('should_continue', False)
                base_dict['next_steps'] = self.output_data.get('next_steps', [])
        
        return base_dict

class IterativeReasoningEngine:
    def __init__(self, reasoning_llm, general_llm, tools: Dict, tools_metadata: Dict, max_iterations: int = 5):
        self.reasoning_llm = reasoning_llm
        self.general_llm = general_llm
        self.tools = tools
        self.tools_metadata = tools_metadata
        self.max_iterations = max_iterations
        self.reasoning_history: List[ReasoningStep] = []
        self.current_context: Dict = {}
        self.original_task: Optional[str] = None
        self.available_tools: List[str] = []
    
    def _format_tools_for_llm(self) -> str:
        tools_desc = []
        for tool in self.tools_metadata.get('tools', []):
            tool_info = f"Tool: {tool['name']}\nDescription: {tool['description']}\nTags: {', '.join(tool['tags'])}\nExamples: {', '.join(tool['examples'])}"
            tools_desc.append(tool_info)
        return "\n\n".join(tools_desc)
    
    def _format_tool_parameters(self, tool_names: list) -> str:
        """Format tool parameter requirements for planning phase"""
        tool_params = []
        for tool_meta in self.tools_metadata.get('tools', []):
            if tool_meta['name'] in tool_names:
                params_desc = f"\n{tool_meta['name']}:"
                if 'parameter_requirements' in tool_meta:
                    params_desc += f"\n  {tool_meta['parameter_requirements']}"
                params_desc += "\n  Parameters:"
                for param_name, param_info in tool_meta.get('parameters', {}).items():
                    required = "REQUIRED" if param_info.get('required', False) else "optional"
                    params_desc += f"\n    - {param_name} ({required}): {param_info.get('description', '')}"
                tool_params.append(params_desc)
        return "\n".join(tool_params) if tool_params else "No specific parameter requirements."
    
    async def _validate_plan(self, plan: Dict) -> Dict:
        """Validate plan and auto-fill missing critical parameters from task context"""
        execution_plan = plan.get('execution_plan', [])
        enhanced_plan = []
        
        for step in execution_plan:
            action = step.get('action', '')
            parameters = step.get('parameters', {})
            
            if action == 'region_mapper':
                if not parameters or (not parameters.get('region_name') and not parameters.get('coordinates')):
                    print(f"[PLAN VALIDATION] region_mapper missing parameters, attempting auto-fill from task context")
                    
                    extraction_prompt = f"""
                    Extract the location or region name from the following task:
                    
                    ORIGINAL TASK: {self.original_task}
                    
                    If the task mentions a specific location, region, city, or geographic area, respond with ONLY a JSON object:
                    {{"region_name": "extracted location name"}}
                    
                    If you find coordinates like [lat, lon] or (lat, lon), respond with:
                    {{"coordinates": [lat, lon]}}
                    
                    If no location is found, respond with:
                    {{"region_name": null}}
                    """
                    
                    response = await self.reasoning_llm.reason(extraction_prompt, show_thinking=False)
                    extracted = self._extract_json(response)
                    
                    if extracted and not extracted.get('error'):
                        if extracted.get('region_name') and extracted['region_name'] is not None:
                            if not parameters:
                                parameters = {}
                            parameters['region_name'] = extracted['region_name']
                            print(f"[PLAN VALIDATION] Retrieved region_name (validation step): {extracted['region_name']}")
                            step['parameters'] = parameters
                        elif extracted.get('coordinates'):
                            if not parameters:
                                parameters = {}
                            parameters['coordinates'] = extracted['coordinates']
                            print(f"[PLAN VALIDATION] Retrieved coordinates (validation step): {extracted['coordinates']}")
                            step['parameters'] = parameters
                        else:
                            print(f"[PLAN VALIDATION] ERROR: Could not extract location from task - user clarification needed")
                            step['validation_error'] = {
                                'error': 'missing_location',
                                'message': 'Cannot determine location for region_mapper. Please specify a region name or coordinates.',
                                'user_input_required': True,
                                'prompt_user': f"The task '{self.original_task}' requires a geographic location but none was specified. Please provide:\n- A region name (e.g., 'Taiwan Strait', 'New York'), OR\n- Coordinates as [latitude, longitude] (e.g., [25.0, 121.5])"
                            }
                    else:
                        print(f"[PLAN VALIDATION] ERROR: Could not parse LLM response for location extraction")
                        step['validation_error'] = {
                            'error': 'extraction_failed',
                            'message': 'Failed to extract location information from task context.',
                            'user_input_required': True,
                            'prompt_user': f"Please specify the geographic location for this task:\n- A region name (e.g., 'Taiwan Strait', 'New York'), OR\n- Coordinates as [latitude, longitude] (e.g., [25.0, 121.5])"
                        }
            
            enhanced_plan.append(step)
        
        plan['execution_plan'] = enhanced_plan
        return plan
    
    
    def _extract_relevant_tools(self, discovery_response: str) -> List[str]:
        try:
            data = self._extract_json(discovery_response)
            if 'relevant_tools' in data:
                return data['relevant_tools']
            if 'tools' in data:
                return data['tools']
            return list(self.tools.keys())
        except:
            return list(self.tools.keys())
    
    def log_external_step(self, step_name: str, input_data: Dict, output_data: Dict, message: str = "", confidence: float = 0.5):
        self.reasoning_history.append(ReasoningStep(
            state=ReasoningState.EXTERNAL,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_data={"step": step_name, **(input_data or {})},
            output_data=output_data or {},
            confidence=confidence,
            reasoning=step_name,
            next_actions=[],
            message=message or ""
        ))
        
    async def reason(self, situation_data: Dict) -> Dict:
        self.current_context = situation_data.copy()
        self.reasoning_history = []
        reasoning_cycle = 0
        loop_iterations = 0
        max_loop_iterations = self.max_iterations * 5
        current_state = ReasoningState.INITIAL
        
        while reasoning_cycle < self.max_iterations and loop_iterations < max_loop_iterations:
            loop_iterations += 1
            
            try:
                if current_state == ReasoningState.INITIAL:
                    current_state = await self._think_phase()
                    reasoning_cycle = 1  # Starting first cycle
                    print(f"[Reasoning Engine] Starting reasoning cycle {reasoning_cycle}")
                    
                elif current_state == ReasoningState.THINKING:
                    current_state = await self._plan_phase()
                    
                elif current_state == ReasoningState.PLANNING:
                    current_state = await self._execute_phase()
                    
                elif current_state == ReasoningState.EXECUTING:
                    current_state = await self._reflect_phase()
                    print(f"[Reasoning Engine] Completed reasoning cycle {reasoning_cycle}")
                    
                elif current_state == ReasoningState.REFLECTING:
                    if await self._should_continue() and reasoning_cycle < self.max_iterations:
                        reasoning_cycle += 1
                        current_state = ReasoningState.THINKING
                        print(f"[Reasoning Engine] Starting reasoning cycle {reasoning_cycle}")
                    else:
                        current_state = ReasoningState.COMPLETED
                        print(f"[Reasoning Engine] Reasoning complete after {reasoning_cycle} cycle(s)")
                        
                elif current_state == ReasoningState.COMPLETED:
                    break
                else:
                    break
                    
            except Exception as e:
                print(f"[Reasoning Engine] ERROR in {current_state}: {e}")
                import traceback
                traceback.print_exc()
                break
                
        if loop_iterations >= max_loop_iterations:
            print(f"[Reasoning Engine] WARNING: Hit maximum loop iterations ({max_loop_iterations})")
                
        return await self._synthesize_final_result()
    
    async def _think_phase(self) -> ReasoningState:
        if not self.original_task:
            self.original_task = self.current_context.get('task_description', '')
        
        historical_context = self.current_context.get('historical_context', {})
        additional_context = self.current_context.get('additional_context', '')
        tools_context = self._format_tools_for_llm()
        
        context_display = f"Task: {self.original_task}"
        if additional_context:
            context_display += f"\n\nAdditional Context: {additional_context}"
        
        think_prompt = f"""
        As a satellite operations reasoning agent, analyze this task comprehensively:

        {context_display}
        
        Available Tools:
        {tools_context}

        Previous Reasoning Steps: {len(self.reasoning_history)}

        Perform complete analysis:
        1. What is being asked? (task understanding)
        2. Which tools from the list above are relevant?
        3. What is the core problem and key risks?
        4. What information do we need to gather?
        5. What are the constraints and requirements?
        6. What is your confidence level (0.0-1.0)?

        Provide your analysis in this EXACT JSON format:
        {{
            "task_understanding": "Clear description of what is being asked",
            "relevant_tools": ["tool1", "tool2", "tool3", "tool4", "tool5", etc.],
            "problem_analysis": "Core issue and approach",
            "risk_assessment": "Key risks and their severity",
            "information_needs": ["list", "of", "needed", "data"],
            "constraints": ["list", "of", "constraints"],
            "confidence": 0.8
        }}

        IMPORTANT: confidence must be a number between 0.0 and 1.0
        """
        
        response = await self.reasoning_llm.reason(think_prompt, show_thinking=True)
        print(f"[THINK PHASE] LLM Response:\n{response}\n")
        
        analysis = await self._extract_json_with_retry(response, self.reasoning_llm, think_prompt, silent_first_fail=True)
        
        if 'error' in analysis:
            print(f"[THINK PHASE] ERROR: All retries failed, using fallback")
            analysis = {
                'task_understanding': f"Task: {self.original_task}",
                'relevant_tools': list(self.tools.keys())[:3],
                'problem_analysis': f"Task: {self.original_task}. Analysis failed after retries.",
                'information_needs': ['satellite imagery', 'detection tools', 'region mapping'],
                'constraints': ['tool availability', 'data quality'],
                'confidence': 0.3,
                'error': 'JSON parsing failed after retries'
            }
        else:
            print(f"[THINK PHASE] Extracted Analysis:")
            print(f"{analysis}")
        
        self.available_tools = self._extract_relevant_tools(response)
        self.current_context['available_tools'] = self.available_tools
        self.current_context['task_understanding'] = analysis.get('task_understanding', '')
        self.current_context['discovered_tools'] = analysis.get('relevant_tools', [])
        
        confidence = self._parse_confidence(analysis.get('confidence', 0.5))
        analysis['original_task'] = self.original_task
        
        step = ReasoningStep(
            state=ReasoningState.THINKING,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_data=self.current_context,
            output_data=analysis,
            confidence=confidence,
            reasoning=analysis.get('task_understanding', ''),
            next_actions=analysis.get('information_needs', []),
            message=response
        )
        
        self.reasoning_history.append(step)
        return ReasoningState.THINKING
    
    async def _plan_phase(self) -> ReasoningState:
        last_analysis = self.reasoning_history[-1].output_data
        available_tools = self.current_context.get('available_tools', list(self.tools.keys()))
        
        original_task = self.original_task or self.current_context.get('task_description', 'Unknown task')
        
        problem_analysis = last_analysis.get('problem_analysis', '')
        if not problem_analysis and 'raw_text' in last_analysis:
            problem_analysis = f"Analysis available but not structured. Original task: {original_task}"
        elif not problem_analysis:
            problem_analysis = f"Task: {original_task}"
        
        additional_context = self.current_context.get('additional_context', '')
        context_info = f"ORIGINAL TASK: {original_task}"
        if additional_context:
            context_info += f"\nADDITIONAL CONTEXT: {additional_context}"
        
        tool_parameters_info = self._format_tool_parameters(available_tools)
        
        plan_prompt = f"""
        Based on your analysis, create an execution plan using available tools:

        {context_info}
        
        Problem Analysis: {problem_analysis}
        Information Needs: {last_analysis.get('information_needs', [])}
        Available Tools: {available_tools}

        TOOL PARAMETER REQUIREMENTS:
        {tool_parameters_info}

        CRITICAL INSTRUCTIONS:
        - Your plan MUST address the ORIGINAL TASK above. Do not invent a different task.
        - You MUST include at least one tool in the execution_plan array.
        - You MUST populate the "parameters" field with appropriate values extracted from the ORIGINAL TASK.
        - DO NOT leave parameters empty {{}} unless the tool requires no parameters.
        - Extract location names, coordinates, or other relevant data from the task context.

        Provide your plan in this EXACT JSON format:
        {{
            "execution_plan": [
                {{"step": 1, "action": "tool_name", "parameters": {{"param1": "value1"}}, "reason": "why this step"}}
            ],
            "expected_outcomes": "What we expect to learn",
            "success_criteria": "How we'll know if this worked",
            "confidence": 0.8
        }}

        IMPORTANT: confidence must be a number between 0.0 and 1.0
        """
        
        response = await self.reasoning_llm.reason(plan_prompt, show_thinking=False)
        print(f"[PLAN PHASE] LLM Response:\n{response}\n")
        
        plan = await self._extract_json_with_retry(response, self.reasoning_llm, plan_prompt)
        
        print(f"[PLAN PHASE] Extracted Plan (before validation):")
        print(f"{plan}")
        
        plan = await self._validate_plan(plan)
        
        print(f"[PLAN PHASE] Enhanced Plan (after validation):")
        print(f"{plan}")
        
        confidence = self._parse_confidence(plan.get('confidence', 0.5))
        
        step = ReasoningStep(
            state=ReasoningState.PLANNING,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_data=last_analysis,
            output_data=plan,
            confidence=confidence,
            reasoning=plan.get('expected_outcomes', ''),
            next_actions=[action['action'] for action in plan.get('execution_plan', [])],
            message=response
        )
        
        self.reasoning_history.append(step)
        return ReasoningState.PLANNING
    
    async def _execute_phase(self) -> ReasoningState:
        last_plan = self.reasoning_history[-1].output_data
        execution_plan = last_plan.get('execution_plan', [])
        
        print(f"\n[EXECUTE PHASE] Execution plan: {json.dumps(execution_plan, indent=2)}")
        print(f"[EXECUTE PHASE] Available tools: {list(self.tools.keys())}")
        
        results = {}
        for step_info in execution_plan:
            action = step_info['action']
            parameters = step_info.get('parameters', {})
            
            if 'validation_error' in step_info:
                validation_error = step_info['validation_error']
                print(f"[EXECUTE PHASE] Step {step_info.get('step')} has validation error: {validation_error['message']}")
                results[f"{action}_step_{step_info['step']}"] = {
                    'status': 'validation_failed',
                    'error': validation_error['error'],
                    'message': validation_error['message'],
                    'user_input_required': validation_error.get('user_input_required', False),
                    'prompt_user': validation_error.get('prompt_user', ''),
                    'tool': action
                }
                continue
            
            print(f"[EXECUTE PHASE] Executing step {step_info.get('step')}: {action} with params {parameters}")
            
            if action in self.tools:
                try:
                    tool_func = self.tools[action]['execute']
                    print(f"[EXECUTE PHASE] Calling tool function: {action}")
                    result = await tool_func(parameters)
                    print(f"[EXECUTE PHASE] Tool {action} result: {result}")
                    results[f"{action}_step_{step_info['step']}"] = result
                except Exception as e:
                    print(f"[EXECUTE PHASE] Tool {action} ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    results[f"{action}_step_{step_info['step']}"] = {'error': str(e)}
            else:
                print(f"[EXECUTE PHASE] Tool '{action}' not found. Available: {list(self.tools.keys())}")
                results[f"{action}_step_{step_info['step']}"] = {'error': f'Tool {action} not found'}
        
        total_actions = len(execution_plan)
        successful_actions = sum(1 for r in results.values() if isinstance(r, dict) and 'error' not in r)
        failed_actions = total_actions - successful_actions
        
        execution_confidence = successful_actions / total_actions if total_actions > 0 else 0.0
        
        execution_summary = f"Executed {len(execution_plan)} actions: {', '.join([s['action'] for s in execution_plan])} ({successful_actions} successful, {failed_actions} failed)"
        print(f"[EXECUTE PHASE] Execution Summary:")
        print(f"{execution_summary}")
        
        step = ReasoningStep(
            state=ReasoningState.EXECUTING,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_data=last_plan,
            output_data={
                'results': results,
                'execution_summary': execution_summary,
                'actions_executed': [s['action'] for s in execution_plan],
                'successful_actions': successful_actions,
                'failed_actions': failed_actions,
                'total_actions': total_actions
            },
            confidence=execution_confidence,
            reasoning=execution_summary,
            next_actions=[],
            message=execution_summary
        )
        
        self.reasoning_history.append(step)
        return ReasoningState.EXECUTING
    
    async def _reflect_phase(self) -> ReasoningState:
        last_execution = self.reasoning_history[-1].output_data
        execution_results = last_execution.get('results', {})
        original_plan = self.reasoning_history[-2].output_data
        
        total_actions = last_execution.get('total_actions', 0)
        successful_actions = last_execution.get('successful_actions', 0)
        
        not_implemented_count = sum(1 for r in execution_results.values() 
                                    if isinstance(r, dict) and r.get('status') == 'not_implemented')
        
        user_input_required = []
        for tool_name, result in execution_results.items():
            if isinstance(result, dict) and result.get('user_input_required', False):
                user_input_required.append({
                    'tool': tool_name,
                    'prompt': result.get('prompt_user', ''),
                    'message': result.get('message', '')
                })
        
        execution_failures = []
        for tool_name, result in execution_results.items():
            if isinstance(result, dict):
                if 'error' in result and result.get('status') not in ['not_implemented', 'validation_failed']:
                    execution_failures.append(f"{tool_name}: {result['error']}")
        
        situation_analysis = []
        
        if user_input_required:
            for req in user_input_required:
                situation_analysis.append(f"🔴 USER INPUT REQUIRED: {req['message']}")
        
        if total_actions > 0 and not_implemented_count == total_actions:
            situation_analysis.append(f"⚠️ All {total_actions} required tools are not yet implemented")
        elif total_actions > 0 and not_implemented_count > 0:
            situation_analysis.append(f"⚠️ {not_implemented_count}/{total_actions} tools are not implemented")
        
        if execution_failures:
            situation_analysis.append(f"⚠️ {len(execution_failures)} tool(s) failed with errors")
        
        if successful_actions == 0 and total_actions > 0:
            situation_analysis.append("⚠️ No tools executed successfully")
        
        situation_context = "\n".join(situation_analysis) if situation_analysis else ""
        
        blocking_guidance = ""
        if user_input_required:
            user_prompts = "\n".join([f"- {req['prompt']}" for req in user_input_required])
            blocking_guidance = f"""
            USER INPUT REQUIRED:
            {situation_context}
            
            The following information is needed from the user:
            {user_prompts}
            
            CRITICAL: Set task_status to "blocked" and should_continue to false.
            Set recommendation to clearly explain what information is needed from the user.
            """
        elif situation_analysis:
            blocking_guidance = f"""
            EXECUTION ANALYSIS:
            {situation_context}
            
            CRITICAL DECISION POINT - Assess if the task can proceed:
            
            Set task_status and should_continue based on YOUR analysis:
            
            → "blocked" + false IF:
            - Required tools/capabilities are missing
            - Insufficient data and no way to acquire it now
            - External dependencies not met (satellites not scheduled, data not available, etc.)
            - Task fundamentally cannot proceed without changes outside your control
            
            → "pending" + false IF:
            - Task requires human intervention or approval
            - Need to wait for scheduled events (satellite pass, data collection, etc.)
            - Partial progress made but requires operator decision
            
            → "failed" + false IF:
            - Tools exist but consistently error
            - Approach is incorrect and alternative not available
            - Critical errors that cannot be resolved
            
            → "pending" + true IF:
            - Some progress made and retrying might help
            - Alternative approaches available to try
            - Recoverable errors that can be fixed with iteration
            """
        else:
            blocking_guidance = """
            EXECUTION SUCCESS - Tools executed successfully.
            
            Evaluate if objectives were met:
            - Set task_status to "completed" if goals achieved
            - Set should_continue to true if more work needed
            - Set should_continue to false if satisfied with results
            """
            
        reflect_prompt = f"""
        Reflect on the execution results and decide next steps:

        Original Plan: {json.dumps(original_plan, indent=2)}
        Execution Results: {json.dumps(execution_results, indent=2)}
        Success Criteria: {original_plan.get('success_criteria', '')}
        
        {blocking_guidance}

        CRITICAL SELF-EVALUATION:
        1. Did we achieve our goals? What was accomplished?
        2. What new information did we learn from the results?
        3. Are there gaps, missing data, or unmet dependencies?
        4. Can this task proceed, or is it blocked by external factors?
        5. Do we need satellite rescheduling, more data, or external actions?
        6. What is your confidence (0.0-1.0) in the current progress?
        7. Should we continue iterating, or stop here?

        Provide reflection in this JSON format:
        {{
            "goal_achievement": "Clear assessment: achieved/not achieved/partial/blocked",
            "new_insights": ["what we learned from execution"],
            "gaps_identified": ["missing data, tools, or capabilities"],
            "blocking_factors": ["external dependencies, if any"],
            "next_steps": ["what should happen next"],
            "task_status": "completed|pending|blocked|failed",
            "confidence": 0.5,
            "should_continue": true or false,
            "recommendation": "Clear recommendation for operators"
        }}

        IMPORTANT: 
        - confidence must be a number between 0.0 and 1.0
        - Be HONEST about blockers - don't continue if fundamentally blocked
        - Set should_continue=false if: tools missing, data unavailable, external dependencies not met
        - Set should_continue=true ONLY if iterating would make meaningful progress
        """
        
        response = await self.reasoning_llm.reason(reflect_prompt, show_thinking=True)
        print(f"[REFLECT PHASE] LLM Response:\n{response}\n")
        
        reflection = await self._extract_json_with_retry(response, self.reasoning_llm, reflect_prompt, silent_first_fail=True)
        
        task_status = reflection.get('task_status', 'pending')
        should_continue = reflection.get('should_continue', False)
        blocking_factors = reflection.get('blocking_factors', [])
        
        print(f"[REFLECT PHASE] Extracted Reflection:")
        print(f"{reflection}")
        
        if blocking_factors:
            print(f"[Reasoning Engine] Blocking factors identified: {', '.join(blocking_factors)}")
        if task_status in ['blocked', 'failed']:
            print(f"[Reasoning Engine] Task status: {task_status}")
        
        confidence = self._parse_confidence(reflection.get('confidence', 0.5))
        
        step = ReasoningStep(
            state=ReasoningState.REFLECTING,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_data=execution_results,
            output_data=reflection,
            confidence=confidence,
            reasoning=reflection.get('recommendation', ''),
            next_actions=reflection.get('next_steps', []),
            message=response
        )
        
        self.reasoning_history.append(step)
        return ReasoningState.REFLECTING
    
    async def _should_continue(self) -> bool:
        if not self.reasoning_history:
            return True
            
        last_reflection = self.reasoning_history[-1].output_data
        should_continue = last_reflection.get('should_continue', False)
        confidence = self._parse_confidence(last_reflection.get('confidence', 0.5))
        task_status = last_reflection.get('task_status', 'pending')
        blocking_factors = last_reflection.get('blocking_factors', [])
        
        if task_status in ['blocked', 'failed', 'impossible']:
            reason = f"task_status={task_status}"
            if blocking_factors:
                reason += f" (blockers: {', '.join(blocking_factors)})"
            print(f"[Reasoning Engine] Stopping - {reason}")
            return False
        
        if not should_continue:
            reason = "LLM decided not to continue"
            if task_status == 'completed':
                reason += " - task completed"
            print(f"[Reasoning Engine] Stopping - {reason} (confidence: {confidence:.2f})")
            return False
        
        if confidence < 0.3:
            print(f"[Reasoning Engine] Continuing - low confidence ({confidence:.2f}), iteration may help")
            return True
        
        if should_continue and confidence < 0.7:
            print(f"[Reasoning Engine] Continuing - LLM requested continuation (confidence: {confidence:.2f})")
            return True
        
        if confidence >= 0.7:
            print(f"[Reasoning Engine] Stopping - sufficient confidence reached ({confidence:.2f})")
            return False
        
        print(f"[Reasoning Engine] Stopping - default (confidence: {confidence:.2f}, continue: {should_continue})")
        return False
    
    async def _synthesize_final_result(self) -> Dict:
        tool_results = {}
        user_input_requests = []
        
        for step in self.reasoning_history:
            if step.state == ReasoningState.EXECUTING and step.output_data:
                tool_results.update(step.output_data)
                results = step.output_data.get('results', {})
                for tool_name, result in results.items():
                    if isinstance(result, dict) and result.get('user_input_required', False):
                        user_input_requests.append({
                            'tool': tool_name,
                            'message': result.get('message', ''),
                            'prompt': result.get('prompt_user', '')
                        })
        
        synthesis_prompt = f"""
        Synthesize a final comprehensive result from this reasoning process:

        Reasoning History: {json.dumps([step.to_dict() for step in self.reasoning_history], indent=2)}

        Tool Results: {json.dumps(tool_results, indent=2, default=str)}

        Create a final operator report that accurately reflects the actual results.

        Provide response in this EXACT JSON format:
        {{
            "situation_summary": "Accurate summary based on actual results",
            "analysis": "Key findings and insights from the reasoning process",
            "recommendations": ["list", "of", "recommendations"],
            "confidence": 0.5,
            "task_status": "completed|pending|failed",
            "reasoning_steps": {len(self.reasoning_history)},
            "next_actions": ["immediate", "next", "steps"],
            "timestamp": "{datetime.utcnow().isoformat()}"
        }}

        IMPORTANT: confidence must be a number between 0.0 and 1.0
        """
        
        response = await self.reasoning_llm.reason(synthesis_prompt, show_thinking=False)
        print(f"[SYNTHESIS] LLM Response:\n{response}\n")
        
        final_result = await self._extract_json_with_retry(response, self.reasoning_llm, synthesis_prompt)
        
        print(f"[SYNTHESIS] Final Result:")
        print(f"  - Summary: {final_result.get('situation_summary', 'N/A')[:200]}...")
        print(f"  - Confidence: {final_result.get('confidence', 0.5)}")
        print(f"  - Task Status: {final_result.get('task_status', 'unknown')}")
        
        final_result['confidence'] = self._parse_confidence(final_result.get('confidence', 0.5))
        final_result['reasoning_trace'] = [step.to_dict() for step in self.reasoning_history]
        final_result['tool_results'] = tool_results
        final_result['total_steps'] = len(self.reasoning_history)
        
        if user_input_requests:
            final_result['user_input_required'] = True
            final_result['user_prompts'] = user_input_requests
        else:
            final_result['user_input_required'] = False
        
        reasoning_iterations = len([s for s in self.reasoning_history if s.state == ReasoningState.THINKING])
        final_result['total_iterations'] = reasoning_iterations
        
        print(f"[Reasoning Engine] Final result: {reasoning_iterations} iteration(s), {len(self.reasoning_history)} total steps, confidence: {final_result['confidence']:.2f}")
        
        return final_result
    
    def _parse_confidence(self, confidence_raw) -> float:
        try:
            if isinstance(confidence_raw, str):
                import re
                numbers = re.findall(r'0\.\d+|\d+\.\d+|\d+', confidence_raw)
                if numbers:
                    return float(numbers[0])
                if "high" in confidence_raw.lower():
                    return 0.8
                elif "medium" in confidence_raw.lower():
                    return 0.6
                elif "low" in confidence_raw.lower():
                    return 0.4
                else:
                    print(f"[Reasoning Engine] ERROR: Could not parse confidence '{confidence_raw}', using default 0.5")
                    return 0.5
            else:
                return float(confidence_raw)
        except (ValueError, TypeError):
            print(f"[Reasoning Engine] ERROR: Invalid confidence value '{confidence_raw}', using default 0.5")
            return 0.5
    
    def _extract_json(self, text: str) -> Dict:
        import re
        
        text_clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text_clean, re.DOTALL)
        if json_block_match:
            try:
                return json.loads(json_block_match.group(1))
            except json.JSONDecodeError:
                pass
        
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text_clean, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        json_match_original = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match_original:
            try:
                return json.loads(json_match_original.group(0))
            except json.JSONDecodeError:
                pass
        
        return {"error": "Could not parse JSON response", "raw_text": text}
    
    async def _extract_json_with_retry(self, text: str, llm, expected_format: str, max_retries: int = 3, silent_first_fail: bool = False) -> Dict:
        result = self._extract_json(text)
        
        if "error" not in result:
            return result
        
        if not silent_first_fail:
            print(f"[JSON Extraction] Initial parse failed, retrying...")
        
        for attempt in range(1, max_retries + 1):
            retry_prompt = f"""
            Your previous response could not be parsed as valid JSON.
            
            Please provide ONLY a valid JSON object in the following format:
            {expected_format}
            
            CRITICAL REQUIREMENTS:
            1. Respond with ONLY the JSON object, no other text
            2. Do NOT wrap it in code blocks or backticks
            3. Ensure all strings use double quotes, not single quotes
            4. Ensure all numbers are valid (no trailing commas)
            5. Confidence values must be numbers between 0.0 and 1.0
            
            Provide the JSON now:
            """
            
            retry_response = await llm.reason(retry_prompt, show_thinking=False)
            result = self._extract_json(retry_response)
            
            # If successful, return
            if "error" not in result:
                return result
        
        # All retries failed
        print(f"[JSON Extraction] ERROR: All {max_retries} retries failed.")
        return result