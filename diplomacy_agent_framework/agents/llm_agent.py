import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
import time
from datetime import datetime

from openai import OpenAI # Use OpenAI client as shown

from diplomacy_agent_framework.core.base_agent import BaseAgent
from diplomacy_agent_framework.core.data_structures import AgentContextBundle, ActionToolCall

# Load environment variables from .env file
load_dotenv()

class LLMAgent(BaseAgent):
    """An agent that uses an LLM via the OpenAI client interface to decide actions."""

    def __init__(self, power_name: str, llm_model_name: str, llm_temperature: float, llm_max_tokens: int):
        super().__init__(power_name)
        # Use OpenRouter configuration - Load API key from environment variable
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        if not self.api_key:
            raise ValueError("OpenRouter API key not found in environment variables (OPENROUTER_API_KEY).")

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        # Use the specified Deepseek model via OpenRouter
        self.model_name = "deepseek/deepseek-r1-zero:free" 
        self.max_tokens = llm_max_tokens # Use passed-in max_tokens
        self.temperature = llm_temperature # Use passed-in temperature
        # Keep original parameters passed to __init__ (Review if needed)
        self.llm_model_name = llm_model_name 
        self.llm_temperature = llm_temperature 
        self.llm_max_tokens = llm_max_tokens 
        self.memory: Dict[str, Any] = {} # Initialize agent memory
        self.log_filename = None # <-- Initialize log filename

    def _format_prompt(self, context_bundle: AgentContextBundle) -> str:
        """Formats the agent context into a string prompt for the LLM."""
        # Get phase info for later use
        phase_info = context_bundle.current_phase_info
        interaction_type = phase_info.interaction_type

        # --- Base Prompt Structure ---
        prompt_lines = []
        prompt_lines.append(f"You are {context_bundle.power_name}, playing Diplomacy.")

        # --- Order Submission Focused Prompt ---
        if interaction_type == "ORDER_SUBMISSION":
            # 1. PRIORITIZE Current State
            prompt_lines.append("\n== CURRENT GAME STATE (Focus!) ==")
            your_units_list = context_bundle.public_game_state.units.get(self.power_name, [])
            your_units_str_simple = ", ".join([f"{u[0]} {u[1]}" for u in your_units_list]) 
            all_units_str = json.dumps(context_bundle.public_game_state.units) 
            centers_str = json.dumps(context_bundle.public_game_state.centers)
            prompt_lines.append(f"  Your Units: {your_units_str_simple}") 
            prompt_lines.append(f"  All Units: {all_units_str}") 
            prompt_lines.append(f"  Supply Centers: {centers_str}")
            if context_bundle.public_game_state.builds_disbands:
                 builds_str = json.dumps(context_bundle.public_game_state.builds_disbands)
                 prompt_lines.append(f"  Builds/Disbands Available: {builds_str}")
            prompt_lines.append("")

            # 2. Current Situation
            prompt_lines.append("== CURRENT SITUATION ==")
            prompt_lines.append(f"Phase: {context_bundle.current_phase_info.phase_name}")
            prompt_lines.append(f"Your Interaction Step: {interaction_type}")

            # 3. Messages & Memory (Still potentially relevant)
            inbox = context_bundle.communication_inbox
            if inbox:
                prompt_lines.append("\n== MESSAGES RECEIVED THIS STEP ==")
                for msg in inbox:
                    prompt_lines.append(f"  From {msg.sender}: {msg.content}")
                prompt_lines.append("")
            memory = context_bundle.private_memory_snippet
            if memory:
                 prompt_lines.append("== YOUR PRIVATE NOTES (Memory) ==")
                 try:
                      memory_str = json.dumps(memory, indent=2)
                      prompt_lines.append(memory_str)
                 except TypeError:
                      prompt_lines.append(str(memory))
                 prompt_lines.append("")

            # 4. Instructions & Tools
            prompt_lines.append("== INSTRUCTIONS & AVAILABLE TOOLS FOR ORDER SUBMISSION ==")
            prompt_lines.append("Your response MUST be a valid JSON list containing allowed tool calls.")
            prompt_lines.append("Pay EXTREME attention to JSON syntax and required arguments.") 
            phase_suffix = context_bundle.current_phase_info.phase_name[-1]
            prompt_lines.append("\nGoal: Submit your final, binding orders for this phase.")
            # Movement Phase Orders
            if phase_suffix == 'M': 
                prompt_lines.append("\nPhase Type: MOVEMENT")
                prompt_lines.append("CRITICAL: Base your orders *only* on the 'Current Game State: Your Units' listed ABOVE.")
                prompt_lines.append("Ensure you issue exactly ONE MOVE, HOLD, or SUPPORT order for EACH of your current units.")
                prompt_lines.append("REQUIRED STEP: Use log_thought FIRST. In its 'thought' argument, provide your step-by-step reasoning: explicitly list your current units, evaluate options for each, explain your final decision, and state the planned single order for EACH unit.")
                prompt_lines.append("\nAllowed Tools for Order Submission (Movement):")
                prompt_lines.append("  - log_thought(thought: str) # Use FIRST. Include detailed reasoning and final plan for ALL units.")
                prompt_lines.append("  - submit_order(order_string: str)")
                prompt_lines.append("      # Format examples: 'A PAR H', 'F MAO - SPA', 'A MAR S A PAR', 'A PIC S F MAO - SPA'...")
            # Adjustment Phase Orders
            elif phase_suffix == 'A': 
                prompt_lines.append("\nPhase Type: ADJUSTMENT")
                prompt_lines.append("CRITICAL: Issue BUILD or DISBAND orders based on 'Builds/Disbands Available', or WAIVE builds. Refer to 'Current Game State'.")
                prompt_lines.append("You MUST account for the exact number of builds (+N) or disbands (-N) required.")
                prompt_lines.append("**If Builds/Disbands count is 0, submit NO orders, just log thought and finish.**")
                prompt_lines.append("REQUIRED STEP: Use log_thought FIRST. In its 'thought' argument, provide reasoning: state your build/disband count, analyze options, and list the specific BUILD/DISBAND/WAIVE orders (or state 'No action needed').")
                prompt_lines.append("\nAllowed Tools for Order Submission (Adjustment):")
                prompt_lines.append("  - log_thought(thought: str) # Use FIRST. Include reasoning and final plan for build/disband/waive (or 'No action').")
                prompt_lines.append("  - submit_order(order_string: str)")
                prompt_lines.append("      # Format examples:")
                prompt_lines.append("      #   Build: 'A PAR B' or 'F BRE B' (MUST be an owned, vacant, HOME supply center)")
                prompt_lines.append("      #   Disband: 'A RUH D' or 'F KIE D'")
                prompt_lines.append("      #   Waive Build: 'WAIVE' (Submit one 'WAIVE' per unused build if builds > 0)")
            # Retreat Phase Orders
            else: # Fallback for Retreat or other unknown phases
                prompt_lines.append("\nPhase Type: RETREAT / OTHER")
                prompt_lines.append("CRITICAL: Submit orders appropriate for this phase type (e.g., Retreat or Disband for units needing retreats). Consult rules.")
                prompt_lines.append("REQUIRED STEP: Use log_thought FIRST. In its 'thought' argument, provide reasoning: list units needing orders, evaluate retreat/disband options, explain choice, and state the planned order for each.")
                prompt_lines.append("\nAllowed Tools for Order Submission (Other):")
                prompt_lines.append("  - log_thought(thought: str) # Use FIRST. Include reasoning and final plan for units needing retreat/disband orders.")
                prompt_lines.append("  - submit_order(order_string: str) # Use appropriate format (e.g., 'A MUN R BER', 'F NAP D').")

            # Common tools/footer for all ORDER_SUBMISSION phases
            prompt_lines.append("\nCommon Tools Available:") 
            prompt_lines.append("  - update_memory(key: str, value: any)")
            prompt_lines.append("\nIMPORTANT: Your response list MUST end with the finish_orders tool call:")
            prompt_lines.append("  - finish_orders()")
            prompt_lines.append("\nExample Final JSON Structure (Movement):") # Simplified example focusing on structure
            prompt_lines.append("```json")
            prompt_lines.append("[")
            prompt_lines.append("  { \"tool_name\": \"log_thought\", \"arguments\": { \"thought\": \"Reasoning: My units are F MAO, A PIC, A MAR. England might move to ENG, so MAO to SPA is risky but necessary for Iberia. PIC support needed. MAR holds. Final Plan: F MAO -> SPA, A PIC S F MAO -> SPA, A MAR H.\" } },")
            prompt_lines.append("  { \"tool_name\": \"submit_order\", \"arguments\": { \"order_string\": \"F MAO - SPA\" } },")
            prompt_lines.append("  { \"tool_name\": \"submit_order\", \"arguments\": { \"order_string\": \"A PIC S F MAO - SPA\" } },")
            prompt_lines.append("  { \"tool_name\": \"submit_order\", \"arguments\": { \"order_string\": \"A MAR H\" } },")
            prompt_lines.append("  { \"tool_name\": \"finish_orders\", \"arguments\": {} }")
            prompt_lines.append("]")
            prompt_lines.append("```")

        # --- Negotiation Focused Prompt (Includes Summary) ---
        elif interaction_type.startswith("NEGOTIATION_ROUND"):
            # 1. Overall Goal
            prompt_lines.append("== OVERALL GOAL ==")
            prompt_lines.append(context_bundle.agent_instructions)
            
            # 2. Game Summary (Included here)
            prompt_lines.append("\n== CURRENT GAME SUMMARY ==")
            prompt_lines.append(context_bundle.history_summary.summary_text if context_bundle.history_summary.summary_text else "The game has just begun.")
            
            # 3. Current Situation & State
            prompt_lines.append("\n== CURRENT SITUATION ==")
            prompt_lines.append(f"Phase: {context_bundle.current_phase_info.phase_name}")
            prompt_lines.append(f"Your Interaction Step: {interaction_type}")
            prompt_lines.append("\n== CURRENT GAME STATE ==")
            your_units_list = context_bundle.public_game_state.units.get(self.power_name, [])
            your_units_str_simple = ", ".join([f"{u[0]} {u[1]}" for u in your_units_list]) 
            all_units_str = json.dumps(context_bundle.public_game_state.units) 
            centers_str = json.dumps(context_bundle.public_game_state.centers)
            prompt_lines.append(f"  Your Units: {your_units_str_simple}") 
            prompt_lines.append(f"  All Units: {all_units_str}") 
            prompt_lines.append(f"  Supply Centers: {centers_str}")
            if context_bundle.public_game_state.builds_disbands:
                 builds_str = json.dumps(context_bundle.public_game_state.builds_disbands)
                 prompt_lines.append(f"  Builds/Disbands Available: {builds_str}")
            prompt_lines.append("")

            # 4. Messages & Memory
            inbox = context_bundle.communication_inbox
            if inbox:
                prompt_lines.append("== MESSAGES RECEIVED THIS STEP ==")
                for msg in inbox:
                    prompt_lines.append(f"  From {msg.sender}: {msg.content}")
                prompt_lines.append("")
            else:
                 prompt_lines.append("== MESSAGES RECEIVED THIS STEP ==")
                 prompt_lines.append("  (None)")
                 prompt_lines.append("")
            memory = context_bundle.private_memory_snippet
            if memory:
                 prompt_lines.append("== YOUR PRIVATE NOTES (Memory) ==")
                 try:
                      memory_str = json.dumps(memory, indent=2)
                      prompt_lines.append(memory_str)
                 except TypeError:
                      prompt_lines.append(str(memory))
                 prompt_lines.append("")

            # 5. Instructions & Tools for Negotiation
            prompt_lines.append("== INSTRUCTIONS & AVAILABLE TOOLS FOR NEGOTIATION ==")
            prompt_lines.append("Your response MUST be a valid JSON list containing allowed tool calls.")
            prompt_lines.append("Pay EXTREME attention to JSON syntax and required arguments.") 
            prompt_lines.append(f"\nGoal: Communicate with other powers, gather info, propose deals, update your memory. Current round: {interaction_type.split('_')[-1]}")
            prompt_lines.append("REQUIRED STEP: Use log_thought FIRST. In its 'thought' argument, provide your reasoning: analyze messages/state, decide communication strategy (who to talk to, what to say/ask), and summarize your planned actions for this round.")
            prompt_lines.append("\nAllowed Tools for Negotiation:")
            prompt_lines.append("  - log_thought(thought: str) # Use FIRST. Include detailed reasoning and plan for this round.")
            prompt_lines.append("  - send_message(recipient: str, message_type: str, content: str) # Recipient MUST be a major power name.")
            prompt_lines.append("  - update_memory(key: str, value: any)")
            prompt_lines.append("\nIMPORTANT: End response list with finish_negotiation_round tool call:")
            prompt_lines.append("  - finish_negotiation_round()")
            prompt_lines.append("\nExample Final JSON Structure (Negotiation):")
            prompt_lines.append("```json")
            prompt_lines.append("[")
            prompt_lines.append("  { \"tool_name\": \"log_thought\", \"arguments\": { \"thought\": \"Reasoning: Need to check England's intentions in ENG. Proposal for DMZ seems safe. Germany is quiet, maybe propose BUR DMZ. Memory: Record proposals. Final Plan: Propose ENG DMZ to ENGLAND, propose BUR DMZ to GERMANY, update memory.\" } },")
            prompt_lines.append("  { \"tool_name\": \"send_message\", \"arguments\": { \"recipient\": \"ENGLAND\", \"message_type\": \"PROPOSAL\", \"content\": \"How about a DMZ in the English Channel this year?\" } },")
            prompt_lines.append("  { \"tool_name\": \"send_message\", \"arguments\": { \"recipient\": \"GERMANY\", \"message_type\": \"PROPOSAL\", \"content\": \"Friendly opening? DMZ Burgundy?\" } },")
            prompt_lines.append("  { \"tool_name\": \"update_memory\", \"arguments\": { \"key\": \"S1901M_prop\", \"value\": \"Proposed DMZ ENG(E), BUR(G)\" } },")
            prompt_lines.append("  { \"tool_name\": \"finish_negotiation_round\", \"arguments\": {} }")
            prompt_lines.append("]")
            prompt_lines.append("```")

        # --- Fallback for Unknown Interaction Type --- 
        else:
            prompt_lines.append("== INSTRUCTIONS ==")
            prompt_lines.append("Unknown interaction type. Use log_thought to explain the situation and then use the appropriate finish tool.")
            prompt_lines.append("Your response MUST be a valid JSON list containing ONLY log_thought and the finish tool.")
            prompt_lines.append("Allowed Tools: log_thought, finish_negotiation_round, finish_orders")
        
        # --- Final Output Instruction (Common to all) --- 
        prompt_lines.append("\nCRITICAL: Generate ONLY the JSON list of tool calls as your response. Do not include any other text, explanations, or formatting like ```json markers before or after the list.")
        prompt_lines.append("Your response MUST start with '[' and end with ']'.")
        prompt_lines.append("\nNow, provide your JSON response:")

        return "\n".join(prompt_lines)

    def _parse_llm_response(self, response_content: str) -> List[ActionToolCall]:
        """Parses the LLM's string response (expected: Reasoning -> Separator -> JSON) into a list of ActionToolCall objects."""
        actions: List[ActionToolCall] = []
        thinking_block = "<Thinking block not found or separated>"
        json_string = ""
        separator = "---JSON RESPONSE---"

        try:
            # Attempt to split the response into thinking and JSON parts
            if separator in response_content:
                parts = response_content.split(separator, 1)
                thinking_block = parts[0].strip()
                json_string = parts[1].strip()
                print(f"DEBUG ({self.power_name}): Found separator. Thinking block extracted ({len(thinking_block)} chars). JSON part starting with: {json_string[:100]}...") # Debug print
            else:
                # Fallback: Assume the whole response is the JSON part (or attempt old parsing)
                print(f"Warning ({self.power_name}): Separator '{separator}' not found in response. Attempting to parse entire response.")
                json_string = response_content.strip()
                thinking_block = "<Separator not found, parsing whole response>"
            
            # --- Clean the potential JSON string --- 
            cleaned_json_string = json_string
            # Clean \boxed{...} wrapper 
            if cleaned_json_string.startswith("\\boxed{"):
                cleaned_json_string = cleaned_json_string[7:].strip()
                if cleaned_json_string.endswith("}"):
                    cleaned_json_string = cleaned_json_string[:-1].strip()
            # Clean markdown fences
            if cleaned_json_string.startswith("```json"):
                cleaned_json_string = cleaned_json_string[7:].strip()
            if cleaned_json_string.startswith("```"):
                 cleaned_json_string = cleaned_json_string[3:].strip()
            if cleaned_json_string.endswith("```"):
                cleaned_json_string = cleaned_json_string[:-3].strip()
            # Final strip
            cleaned_json_string = cleaned_json_string.strip()

            # --- Parse the cleaned JSON string --- 
            if not cleaned_json_string:
                 raise ValueError("Cleaned JSON string is empty after splitting and cleaning.")

            parsed_response = json.loads(cleaned_json_string)
            
            if not isinstance(parsed_response, list):
                raise ValueError("Parsed response is not a JSON list.")

            valid_tools = ActionToolCall.__annotations__['tool_name'].__args__
            for item in parsed_response:
                if isinstance(item, dict) and 'tool_name' in item:
                    tool_name = item.get('tool_name')
                    if tool_name not in valid_tools:
                        print(f"Warning: Invalid tool name '{tool_name}' in LLM response. Skipping.")
                        continue
                    arguments = item.get('arguments', {})
                    if not isinstance(arguments, dict):
                         print(f"Warning: Invalid arguments format for tool {tool_name}. Skipping.")
                         continue
                    actions.append(ActionToolCall(tool_name=tool_name, arguments=arguments))
                else:
                    print(f"Warning: Invalid item format in LLM response list: {item}. Skipping.")
            
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to decode LLM JSON part for {self.power_name}: {e}")
            print(f"Raw response slice near error: ///{response_content[max(0,e.pos-20):e.pos+20]}///") # Show slice near error
            print(f"Cleaned JSON part tried: ///{cleaned_json_string[:500]}...///") # Show cleaned JSON part
            print(f"Thinking Block: ///{thinking_block[:500]}...///") # Show thinking block
            actions = [] 
        except ValueError as e:
            print(f"ERROR: Invalid LLM response structure or content for {self.power_name}: {e}")
            print(f"Cleaned JSON part tried: ///{cleaned_json_string[:500]}...///")
            print(f"Thinking Block: ///{thinking_block[:500]}...///")
            actions = []
        except Exception as e:
             print(f"ERROR: Unexpected error parsing LLM response for {self.power_name}: {e}")
             print(f"Thinking Block: ///{thinking_block[:500]}...///")
             actions = []

        # Optionally, do something with thinking_block here (e.g., log it)
        # For now, it's just printed in the debug/error messages above.

        return actions

    def take_turn(self, context_bundle: AgentContextBundle, run_log_dir: str = None) -> List[ActionToolCall]:
        """Generates actions by calling the LLM and parsing its response."""
        prompt = self._format_prompt(context_bundle)
        max_retries = 3
        retry_delay = 2 # seconds
        
        # --- Generalized Agent Logging --- 
        agent_log_file = None
        if run_log_dir:
            try:
                agent_log_file = os.path.join(run_log_dir, f"{self.power_name}.log")
                # Log context before API call
                with open(agent_log_file, "a", encoding="utf-8") as f:
                    phase_info = context_bundle.current_phase_info
                    f.write(f"\n======================================================================\n")
                    f.write(f"--- {self.power_name} Context [{phase_info.interaction_type}] ({phase_info.phase_name}) ---\n")
                    f.write(f"======================================================================\n")
                    f.write(prompt)
                    f.write(f"\n----------------------------------------------------------------------\n")
            except Exception as log_e:
                 print(f"ERROR: Failed to write {self.power_name} context to log: {log_e}")
                 agent_log_file = None # Disable logging for this agent if context write fails
        # --- End Logging Setup --- 

        actions = []
        response_content = "<LLM Call Failed or Not Executed>"
        
        for attempt in range(max_retries):
            try:
                # Use the framework logger for this message now
                # print(f"LLMAgent ({self.power_name}): Sending request to LLM ({context_bundle.current_phase_info.interaction_type})... Attempt {attempt + 1}/{max_retries}")
                chat_completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stream=False
                )
                
                response_content = chat_completion.choices[0].message.content
                
                if not response_content or response_content.strip() == "":
                    # Use framework logger potentially?
                    # print(f"Warning ({self.power_name}): LLM returned empty response on attempt {attempt + 1}.")
                    raise ValueError("LLM returned empty response")

                # print(f"LLMAgent ({self.power_name}): Received response. Parsing..." )
                
                actions = self._parse_llm_response(response_content)
                
                # No need to extract thinking block here anymore, full response logged below
                
                break # Success

            except Exception as e:
                # Log error via framework logger potentially?
                # print(f"ERROR ({self.power_name}): LLM API call/parsing failed on attempt {attempt + 1}: {e}")
                response_content = f"<LLM API Call/Parse Error Attempt {attempt+1}: {e}>" 
                actions = [] 
                if attempt < max_retries - 1:
                    # print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    # print(f"ERROR ({self.power_name}): Max retries ({max_retries}) reached. Giving up.")
                    response_content = f"<LLM API Call Failed after {max_retries} attempts. Last Error: {e}>"
        
        # --- Log Agent Response --- 
        if agent_log_file: # Log only if file was successfully opened earlier
            try:
                with open(agent_log_file, "a", encoding="utf-8") as f:
                    phase_info = context_bundle.current_phase_info # Get phase info again
                    # Log the final raw response (or error message)
                    f.write(f"--- {self.power_name} Raw Response [{phase_info.interaction_type}] ({phase_info.phase_name}) ---\n")
                    f.write(response_content)
                    f.write(f"\n======================================================================\n")
            except Exception as log_e:
                 print(f"ERROR: Failed to write {self.power_name} response to log: {log_e}")
        # --- End Logging --- 

        # Fallback / Ensure Finish Action
        if not actions or not any(a.tool_name in ["finish_negotiation_round", "finish_orders"] for a in actions):
             # Log fallback to the *agent's* log file
             if agent_log_file:
                  try:
                      with open(agent_log_file, "a", encoding="utf-8") as f:
                           f.write(f"\n--- {self.power_name} Fallback Action [{context_bundle.current_phase_info.interaction_type}] ({context_bundle.current_phase_info.phase_name}) ---\n")
                           f.write("Agent failed to produce valid/complete actions. Using default finish.\n")
                           f.write(f"--- Raw Response Leading to Fallback ---\n{response_content}\n") 
                           f.write(f"======================================================================\n")
                  except Exception as log_e:
                       print(f"ERROR: Failed to write {self.power_name} fallback info to log: {log_e}")

             # Use framework logger for this warning
             # print(f"Warning: LLM Agent {self.power_name} did not produce valid/complete actions (after retries). Using default finish action.")
             interaction_type = context_bundle.current_phase_info.interaction_type
             finish_tool = "finish_negotiation_round" if interaction_type.startswith("NEGOTIATION") else "finish_orders"
             actions = [ActionToolCall(tool_name=finish_tool)]

        return actions 

    @staticmethod
    def _format_summary_prompt(
        previous_summary: str,
        phase_completed: str,
        recent_events: List[str],
        order_results: Dict[str, List[str]]
    ) -> str:
        """Creates the prompt for the LLM to summarize the last phase.
        
        Args:
            previous_summary: The summary narrative up to the start of the phase.
            phase_completed: The name of the phase that just finished (e.g., S1901M).
            recent_events: List of significant events from the completed phase (e.g., captures).
            order_results: Dictionary mapping power name to their order results from the completed phase.

        Returns:
            The prompt string for the LLM summarizer.
        """
        prompt_lines = []
        prompt_lines.append("You are a concise historian summarizing a game of Diplomacy.")
        prompt_lines.append(f"Update the existing game summary with the events of the most recently completed phase ({phase_completed}).")
        prompt_lines.append("Provide a narrative summary in approximately TWO paragraphs.")
        prompt_lines.append("First paragraph: Briefly describe the key strategic outcomes of the phase (e.g., significant captures, major standoffs/bounces, successful coordinated moves). Mention which powers gained or lost centers.")
        prompt_lines.append("Second paragraph: Briefly describe the overall strategic situation - mention major alliances or conflicts that seem to be developing based on the phase results and previous summary.")
        prompt_lines.append("Keep the tone objective and focus on strategically relevant information.")

        prompt_lines.append("\n== PREVIOUS SUMMARY ==")
        prompt_lines.append(previous_summary if previous_summary else "The game has just begun.")
        
        prompt_lines.append(f"\n== INFORMATION FROM PHASE: {phase_completed} ==")
        if recent_events:
             prompt_lines.append("Significant Events (e.g., captures):")
             for event in recent_events:
                 prompt_lines.append(f"  - {event}")
        else:
             prompt_lines.append("No major captures or dislodges occurred this phase.")

        if order_results:
             prompt_lines.append("\nOrder Results Summary:")
             # More detailed summary of order results
             for power, results in order_results.items():
                 success_count = sum(1 for r in results if "Success" in r)
                 fail_count = len(results) - success_count
                 # Example: Get failed orders for more context
                 failed_orders = [r for r in results if "Success" not in r and "Invalid" not in r] # Exclude simple invalid ones
                 failed_str = f", Failures: {len(failed_orders)}" if failed_orders else ""
                 if results: 
                     summary = f"{power}: {success_count} successful{failed_str}"
                     # Optionally add details of key bounces if available in results
                     # Example: if any("Bounced" in r for r in failed_orders): summary += " (incl. bounces)"
                     prompt_lines.append(f"  - {summary}")
        
        prompt_lines.append("\n== INSTRUCTIONS ==")
        prompt_lines.append("Provide the updated summary narrative (approx. two paragraphs). Output ONLY the summary text.")
        
        return "\n".join(prompt_lines)

    @staticmethod
    def summarize_history(previous_summary: str, recent_events: List[str], phase_completed: str, 
                          client: OpenAI, # <-- Add client instance
                          model_name: str, # <-- Add model name
                          temperature: float, # <-- Add temperature
                          max_tokens: int, # <-- Add max tokens
                          ) -> str:
        """Generates a concise summary of game progress using an LLM via the provided client."""
        print(f"LLMAgent (Summarizer): Generating summary for end of {phase_completed}...")
        
        # Combine recent events into a readable string
        events_string = "\n".join(f"- {event}" for event in recent_events) if recent_events else "- No major captures or dislodges detected."

        # Construct the prompt
        summary_prompt = f"""
Given the previous summary and recent events, provide an updated, concise game summary (1-2 sentences max). 
Focus on significant changes like captured centers or major power shifts. Mention the phase just completed ({phase_completed}).

Previous Summary:
{previous_summary}

Recent Events during {phase_completed}:
{events_string}

Updated Summary:"""

        try:
            # Use the passed-in client and config
            chat_completion = client.chat.completions.create(
                model=model_name, 
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=max_tokens, 
                temperature=temperature, 
                stream=False
            )
            updated_summary = chat_completion.choices[0].message.content.strip()
            print("LLMAgent (Summarizer): Summary generated.")
            # Basic validation/fallback
            if not updated_summary or len(updated_summary) < 10:
                print("LLMAgent (Summarizer): Warning - LLM generated empty or very short summary. Falling back.")
                # Fallback to a simple combination if LLM fails
                return previous_summary + f"\n\nDuring {phase_completed}: {events_string}." 
            return updated_summary

        except Exception as e:
            print(f"ERROR: LLM Summary generation failed: {e}")
            # Fallback to a simple combination if LLM fails
            return previous_summary + f"\n\nDuring {phase_completed}: {events_string}." 

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Sends the prompt to the configured LLM and returns the parsed JSON response."""
        # Placeholder for actual LLM API call
        # This method should be implemented to actually call the LLM and return the parsed response
        # For now, it's left empty as it's not used in the current implementation
        return {} 