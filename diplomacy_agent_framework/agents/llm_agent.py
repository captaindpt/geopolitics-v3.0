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
        self.api_key = os.getenv("HF_API_KEY")
        self.base_url = "https://vmjps1ofbtvn2w43.us-east-1.aws.endpoints.huggingface.cloud/v1/"
        if not self.api_key:
            raise ValueError("Hugging Face API key not found in environment variables (HF_API_KEY).")
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        self.model_name = "tgi" # As specified in the example
        self.max_tokens = 1024 
        # Lower temperature for more deterministic, instruction-following behavior
        self.temperature = 0.1 
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
        prompt_lines.append(f"You are {context_bundle.power_name}.")

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
            # Keep memory section brief or remove if too distracting?
            memory = context_bundle.private_memory_snippet
            if memory:
                 prompt_lines.append("== YOUR PRIVATE NOTES (Memory) ==")
                 try:
                      memory_str = json.dumps(memory, indent=2)
                      prompt_lines.append(memory_str)
                 except TypeError:
                      prompt_lines.append(str(memory))
                 prompt_lines.append("")

            # 4. Instructions & Tools (No Summary Here)
            prompt_lines.append("== INSTRUCTIONS & AVAILABLE TOOLS FOR ORDER SUBMISSION ==")
            prompt_lines.append("Your response MUST be a valid JSON list containing allowed tool calls.")
            prompt_lines.append("Pay EXTREME attention to JSON syntax and required arguments.") 
            phase_suffix = context_bundle.current_phase_info.phase_name[-1]
            prompt_lines.append("\nGoal: Submit your final, binding orders for this phase.")
            # Movement Phase Orders
            if phase_suffix == 'M': 
                prompt_lines.append("\nPhase Type: MOVEMENT")
                prompt_lines.append("CRITICAL: Base your orders *only* on the 'Current Game State: Your Units' listed ABOVE.") # Emphasize location
                prompt_lines.append("Ensure you issue exactly ONE MOVE, HOLD, or SUPPORT order for EACH of your current units.")
                prompt_lines.append("REQUIRED STEP: Use log_thought FIRST to explicitly list your current units based on the state ABOVE, then state the single order planned for each unit.")
                prompt_lines.append("\nAllowed Tools for Order Submission (Movement):")
                prompt_lines.append("  - log_thought(thought: str) # Use FIRST to list current units and planned MOVEMENT orders.")
                prompt_lines.append("  - submit_order(order_string: str)")
                prompt_lines.append("      # Format examples: 'A PAR H', 'F MAO - SPA', 'A MAR S A PAR', 'A PIC S F MAO - SPA'...")
                prompt_lines.append("\nExample Movement Order Response:") # Keep example
                prompt_lines.append("```json") # Example start
                prompt_lines.append("[")
                prompt_lines.append("  { \"tool_name\": \"log_thought\", \"arguments\": { \"thought\": \"My current units are: F MAO, A PIC, A MAR. Plan: F MAO -> SPA, A PIC S F MAO -> SPA, A MAR H.\" } },")
                prompt_lines.append("  { \"tool_name\": \"submit_order\", \"arguments\": { \"order_string\": \"F MAO - SPA\" } },")
                prompt_lines.append("  { \"tool_name\": \"submit_order\", \"arguments\": { \"order_string\": \"A PIC S F MAO - SPA\" } },")
                prompt_lines.append("  { \"tool_name\": \"submit_order\", \"arguments\": { \"order_string\": \"A MAR H\" } },")
                prompt_lines.append("  { \"tool_name\": \"finish_orders\", \"arguments\": {} }")
                prompt_lines.append("]")
                prompt_lines.append("```") # Example end
            
            # Adjustment Phase Orders
            elif phase_suffix == 'A': 
                prompt_lines.append("\nPhase Type: ADJUSTMENT")
                prompt_lines.append("CRITICAL: Issue BUILD or DISBAND orders based on 'Builds/Disbands Available' listed ABOVE, or WAIVE builds.")
                prompt_lines.append("Refer to 'Current Game State' for owned Centers (build spots) and Units (for disbands).")
                prompt_lines.append("You MUST account for the exact number of builds (+N) or disbands (-N) required.")
                prompt_lines.append("**If Builds/Disbands count is 0, submit NO orders, just log thought and finish.**") # Added explicit instruction for 0 case
                prompt_lines.append("REQUIRED STEP: Use log_thought FIRST to state your build/disband count and list the specific BUILD/DISBAND/WAIVE orders (or state 'No action needed').")
                prompt_lines.append("\nAllowed Tools for Order Submission (Adjustment):")
                prompt_lines.append("  - log_thought(thought: str) # Use FIRST to state build/disband count and planned ADJUSTMENT orders (or 'No action').")
                prompt_lines.append("  - submit_order(order_string: str)")
                prompt_lines.append("      # Format examples:")
                prompt_lines.append("      #   Build: 'A PAR B' or 'F BRE B' (MUST be an owned, vacant, HOME supply center)") # Emphasized Home SC
                prompt_lines.append("      #   Disband: 'A RUH D' or 'F KIE D'")
                prompt_lines.append("      #   Waive Build: 'WAIVE' (Submit one 'WAIVE' per unused build if builds > 0)")
                prompt_lines.append("      # DO NOT submit Move/Hold/Support orders.")
                prompt_lines.append("\nExample Adjustment Order Response (Build=1):")
                prompt_lines.append("```json")
                prompt_lines.append("[")
                prompt_lines.append("  { \"tool_name\": \"log_thought\", \"arguments\": { \"thought\": \"Builds/Disbands: +1. Plan: Build Army in Paris.\" } },")
                prompt_lines.append("  { \"tool_name\": \"submit_order\", \"arguments\": { \"order_string\": \"A PAR B\" } },")
                prompt_lines.append("  { \"tool_name\": \"finish_orders\", \"arguments\": {} }")
                prompt_lines.append("]")
                prompt_lines.append("```")
                prompt_lines.append("\nExample Adjustment Order Response (Builds=0):") # Added example for 0 builds
                prompt_lines.append("```json")
                prompt_lines.append("[")
                prompt_lines.append("  { \"tool_name\": \"log_thought\", \"arguments\": { \"thought\": \"Builds/Disbands: 0. No action needed.\" } },") 
                prompt_lines.append("  { \"tool_name\": \"finish_orders\", \"arguments\": {} }") # Note: No submit_order calls
                prompt_lines.append("]")
                prompt_lines.append("```")
                # ... (Maybe add Disband/Waive examples too if space allows) ...
                prompt_lines.append("\nExample Adjustment Order Response (Disband=1):") 
                prompt_lines.append("```json")

            # TODO: Handle Retreat Phase (R) Specifics
            else: # Fallback for Retreat or other unknown phases
                prompt_lines.append("\nPhase Type: RETREAT / OTHER")
                prompt_lines.append("CRITICAL: Submit orders appropriate for this phase type (e.g., Retreat or Disband for units needing retreats). Consult rules.")
                prompt_lines.append("REQUIRED STEP: Before submitting orders, use log_thought to list units needing orders and the planned order for each.")
                prompt_lines.append("\nAllowed Tools for Order Submission (Other):")
                prompt_lines.append("  - log_thought(thought: str) # Use this FIRST to list units and planned orders for this phase.")
                prompt_lines.append("  - submit_order(order_string: str) # Use appropriate format (e.g., 'A MUN R BER', 'F NAP D').")

            # Common tools/footer for all ORDER_SUBMISSION phases
            prompt_lines.append("\nCommon Tools Available:") 
            prompt_lines.append("  - update_memory(key: str, value: any)")
            prompt_lines.append("\nIMPORTANT: Your response list MUST end with the finish_orders tool call:")
            prompt_lines.append("  - finish_orders()")

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
            prompt_lines.append("\nAllowed Tools for Negotiation:")
            prompt_lines.append("  - log_thought(thought: str) # Optional: Log reasoning.")
            prompt_lines.append("  - send_message(recipient: str, message_type: str, content: str) # Recipient is POWER name.")
            prompt_lines.append("      # CRITICAL: The 'recipient' MUST be one of the 7 major powers: ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY'].")
            prompt_lines.append("      # DO NOT use any other names or abbreviations like 'Spa' or 'Por'.")
            prompt_lines.append("  - update_memory(key: str, value: any)")
            prompt_lines.append("\nIMPORTANT: End response list with finish_negotiation_round tool call:")
            prompt_lines.append("  - finish_negotiation_round()")
            prompt_lines.append("\nExample Negotiation Response:")
            prompt_lines.append("```json") # Example start
            prompt_lines.append("[")
            prompt_lines.append("  { \"tool_name\": \"log_thought\", \"arguments\": { \"thought\": \"Proposing DMZ in ENG to England.\" } },")
            prompt_lines.append("  { \"tool_name\": \"send_message\", \"arguments\": { \"recipient\": \"ENGLAND\", \"message_type\": \"PROPOSAL\", \"content\": \"DMZ in ENG?\" } },")
            prompt_lines.append("  { \"tool_name\": \"update_memory\", \"arguments\": { \"key\": \"england_prop\", \"value\": \"DMZ ENG\" } },")
            prompt_lines.append("  { \"tool_name\": \"finish_negotiation_round\", \"arguments\": {} }")
            prompt_lines.append("]")
            prompt_lines.append("```") # Example end

        # --- Fallback for Unknown Interaction Type --- 
        else:
            # Keep a very simple fallback
            prompt_lines.append("== INSTRUCTIONS ==")
            prompt_lines.append("Unknown interaction type. Provide minimal response and finish.")
            prompt_lines.append("Allowed Tools: log_thought, finish_negotiation_round, finish_orders")
        
        # --- Final Output Instruction (Common to all) --- 
        prompt_lines.append("\nCRITICAL: Generate ONLY the JSON list of tool calls as your response. Do not include any other text, explanations, or formatting like ```json markers before or after the list.")
        prompt_lines.append("Your response MUST start with '[' and end with ']'.")
        prompt_lines.append("\nNow, provide your JSON response:") # Slightly simplified final line

        return "\n".join(prompt_lines)

    def _parse_llm_response(self, response_content: str) -> List[ActionToolCall]:
        """Parses the LLM's string response into a list of ActionToolCall objects."""
        actions: List[ActionToolCall] = []
        try:
            cleaned_content = response_content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:].strip()
            if cleaned_content.startswith("```"):
                 cleaned_content = cleaned_content[3:].strip()
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3].strip()
            
            parsed_response = json.loads(cleaned_content)
            
            if not isinstance(parsed_response, list):
                raise ValueError("LLM response is not a JSON list.")

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
            print(f"ERROR: Failed to decode LLM JSON response for {self.power_name}: {e}")
            print(f"Raw response: ///{response_content}///")
            actions = [] 
        except ValueError as e:
            print(f"ERROR: Invalid LLM response structure for {self.power_name}: {e}")
            print(f"Cleaned response: ///{cleaned_content}///")
            actions = []
        except Exception as e:
             print(f"ERROR: Unexpected error parsing LLM response for {self.power_name}: {e}")
             actions = []

        return actions

    def take_turn(self, context_bundle: AgentContextBundle) -> List[ActionToolCall]:
        """Generates actions by calling the LLM and parsing its response."""
        prompt = self._format_prompt(context_bundle)
        
        # --- Logging for France --- 
        if self.power_name == "FRANCE":
            if self.log_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.log_filename = f"france_llm_log_{timestamp}.txt"

            log_filename = self.log_filename # Use the instance variable
            phase_info = context_bundle.current_phase_info
            try:
                with open(log_filename, "a", encoding="utf-8") as f: # Use append mode
                    f.write(f"\n======================================================================\n")
                    f.write(f"--- FRANCE Context [{phase_info.interaction_type}] ({phase_info.phase_name}) ---\n") # Corrected f-string
                    f.write(f"======================================================================\n")
                    f.write(prompt)
                    f.write(f"\n----------------------------------------------------------------------\n")
            except Exception as log_e:
                 print(f"ERROR: Failed to write France context to log: {log_e}")
        # --- End Logging --- 

        actions = []
        response_content = "<LLM Call Failed or Not Executed>"
        try:
            print(f"LLMAgent ({self.power_name}): Sending request to LLM ({context_bundle.current_phase_info.interaction_type})...")
            chat_completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature, # Use updated temperature
                stream=False
            )
            
            response_content = chat_completion.choices[0].message.content
            print(f"LLMAgent ({self.power_name}): Received response. Parsing..." )
            # print(f"--- LLMAgent ({self.power_name}) Raw Response ---\n{response_content}\n----------------------------------") 

            actions = self._parse_llm_response(response_content)

        except Exception as e:
            print(f"ERROR: LLM API call failed for {self.power_name}: {e}")
            response_content = f"<LLM API Call Error: {e}>"
            actions = [] 
        
        # --- Logging for France (Response) --- 
        if self.power_name == "FRANCE" and self.log_filename: # Check if filename exists
            log_filename = self.log_filename # Use the instance variable
            phase_info = context_bundle.current_phase_info # Redefine phase_info here just in case
            try:
                with open(log_filename, "a", encoding="utf-8") as f: # Use append mode
                    f.write(f"--- FRANCE Raw Response [{phase_info.interaction_type}] ({phase_info.phase_name}) ---\n") # Corrected f-string
                    f.write(response_content)
                    f.write(f"\n======================================================================\n")
            except Exception as log_e:
                 print(f"ERROR: Failed to write France response to log: {log_e}")
        # --- End Logging --- 

        # Fallback / Ensure Finish Action
        if not actions or not any(a.tool_name in ["finish_negotiation_round", "finish_orders"] for a in actions):
             # Log if we are falling back for France
             if self.power_name == "FRANCE" and self.log_filename: # Check if filename exists
                  log_filename = self.log_filename # Use the instance variable
                  with open(log_filename, "a", encoding="utf-8") as f: # Use append mode
                       f.write(f"\n--- FRANCE Fallback Action [{context_bundle.current_phase_info.interaction_type}] ({context_bundle.current_phase_info.phase_name}) ---\n") # Corrected f-string
                       f.write("Agent failed to produce valid/complete actions. Using default finish.\n")
                       f.write(f"======================================================================\n")

             print(f"Warning: LLM Agent {self.power_name} did not produce valid/complete actions. Using default finish action.")
             interaction_type = context_bundle.current_phase_info.interaction_type
             # Determine correct finish tool based on the *intended* interaction type
             finish_tool = "finish_negotiation_round" if interaction_type.startswith("NEGOTIATION") else "finish_orders"
             # Ensure actions list is replaced, not appended to
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
            recent_events: List of significant events from the completed phase.
            order_results: Dictionary mapping power name to their order results from the completed phase.

        Returns:
            The prompt string for the LLM summarizer.
        """
        prompt_lines = []
        prompt_lines.append("You are a concise historian summarizing a game of Diplomacy.")
        prompt_lines.append("Your goal is to update the existing game summary with the events of the most recently completed phase.")
        prompt_lines.append("Focus on strategically important outcomes: captures, key failed orders (bounces, cuts), and potentially significant negotiations if provided.")
        prompt_lines.append("Keep the summary narrative brief and objective.")
        prompt_lines.append("\n== PREVIOUS SUMMARY ==")
        prompt_lines.append(previous_summary if previous_summary else "The game has just begun.")
        prompt_lines.append(f"\n== EVENTS OF PHASE: {phase_completed} ==")
        if recent_events:
             prompt_lines.append("Significant Events:")
             for event in recent_events:
                 prompt_lines.append(f"  - {event}")
        else:
             prompt_lines.append("No major captures or dislodges occurred.")

        if order_results:
             prompt_lines.append("\nOrder Results Summary:")
             for power, results in order_results.items():
                 # Summarize results briefly, maybe highlight failures?
                 success_count = sum(1 for r in results if "Success" in r)
                 fail_count = len(results) - success_count
                 if results: # Only mention powers that had orders
                     summary = f"{power}: {success_count} successful, {fail_count} failed/bounced orders."
                     # Optionally, add more detail about specific failures if needed
                     prompt_lines.append(f"  - {summary}")
        
        prompt_lines.append("\n== INSTRUCTIONS ==")
        prompt_lines.append("Provide an updated summary incorporating the events of the completed phase into the narrative.")
        prompt_lines.append("Output ONLY the updated summary text, nothing else.")
        
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