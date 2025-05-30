import time
import os
import datetime
import json
from typing import List, Dict, Any, Type, Optional
from dotenv import load_dotenv
from collections import OrderedDict

from .engine_adapter import EngineAdapter
from .base_agent import BaseAgent
from .data_structures import (AgentContextBundle, ActionToolCall, Message,
                              PublicGameState, PhaseInfo, HistorySummary, PowerName)
# Import the LLMAgent to access the static summarization method
from diplomacy_agent_framework.agents.llm_agent import LLMAgent
# We'll need the simple agent for default initialization
from diplomacy_agent_framework.agents.hold_agent import HoldAgent

class FrameworkOrchestrator:
    """Manages the game loop, agent interactions, and communication with the engine."""

    def __init__(
        self,
        agent_classes: Dict[str, Type[BaseAgent]] = None,
        num_negotiation_rounds: int = 2,
        map_name: str = 'standard',
        agent_instances: Dict[str, BaseAgent] = None,
        run_log_dir: str = None
    ):
        """
        Initializes the framework.

        Args:
            agent_classes: A dictionary mapping power names to agent classes.
                           If None, defaults to HoldAgent for all powers.
            num_negotiation_rounds: The number of negotiation rounds per phase.
            map_name: The map to use for the Diplomacy game.
            agent_instances: A dictionary mapping power names to agent instances.
            run_log_dir: The directory to use for logging.
        """
        print("Framework: Initializing Orchestrator...")
        self.engine_adapter = EngineAdapter(map_name=map_name)
        self.powers: List[PowerName] = self.engine_adapter.get_all_powers()
        self.num_negotiation_rounds = num_negotiation_rounds

        print("Framework: Initializing agents...")
        self.agents: Dict[PowerName, BaseAgent] = {}
        if agent_classes is None and agent_instances is None:
            agent_classes = {power: HoldAgent for power in self.powers}

        if agent_instances is None:
            for power_name in self.powers:
                agent_class = agent_classes.get(power_name, HoldAgent) # Default to HoldAgent if missing
                self.agents[power_name] = agent_class(power_name)
                print(f"  - {power_name}: {agent_class.__name__}")
        else:
            self.agents = agent_instances

        # State variables for the current turn
        self._message_staging: Dict[PowerName, List[Message]] = {p: [] for p in self.powers} # Messages waiting for delivery next round/phase
        self._current_orders: Dict[PowerName, List[str]] = {p: [] for p in self.powers} # Orders collected this phase
        # Use OrderedDict for memory with a size limit
        self.memory_limit = 30
        self._agent_memory: Dict[PowerName, OrderedDict[str, Any]] = {p: OrderedDict() for p in self.powers}
        self._previous_public_state: Optional[PublicGameState] = None # Store state from previous phase
        self._all_messages_log: List[Message] = [] # Log all messages for potential future summary
        self._last_phase_results_by_power: Dict[PowerName, List[str]] = {} # Store formatted order results from previous phase
        self._current_history_summary: str = "" # Initialize empty summary string

        # Store LLM config if we need it for summarization (assuming one config for all LLM agents)
        # Find the first LLMAgent to get its config - slightly hacky
        self._llm_config = {"model": "gemini-1.5-flash-latest", "temp": 0.1, "max_tokens": 1500}
        for agent in self.agents.values():
            if isinstance(agent, LLMAgent):
                self._llm_config = {
                    "model": agent.llm_model_name,
                    "temp": agent.llm_temperature, # Might want a different temp for summary
                    "max_tokens": agent.llm_max_tokens
                }
                break # Found one

        # --- Consolidated Logging Setup --- 
        self.run_log_dir = run_log_dir
        self.run_log_file = None
        if self.run_log_dir:
            self.run_log_file = os.path.join(self.run_log_dir, "run.log")
            try:
                # Clear/Create the single log file
                with open(self.run_log_file, 'w', encoding='utf-8') as f:
                    f.write(f"Run Log Initialized: {datetime.datetime.now()}\n")
                self._log_message("Framework: Consolidated run log initialized.") # Log initialization event
            except Exception as e:
                print(f"ERROR: Could not initialize run log file {self.run_log_file}: {e}")
                self.run_log_file = None # Disable logging if file creation fails
        else:
            print("Warning: No run log directory provided. Logging disabled.")
        # --- End Logging Setup ---

        self._log_message("Framework: Initialization complete.")

    def _log_message(self, message: str):
        """Logs a message to the console and the consolidated run log file."""
        print(message) # Keep console output
        if self.run_log_file:
            try:
                with open(self.run_log_file, 'a', encoding='utf-8') as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    # Indent framework messages slightly for readability
                    if message.startswith("    - Framework:"):
                        indent = "  "
                    elif message.startswith("  -- Framework:"):
                         indent = " "
                    elif message.startswith("--- Framework:") or message.startswith("=== Framework:") or message.startswith("Framework:"):
                         indent = ""
                    else: # Default indentation for other messages (like thoughts, errors)
                         indent = "    "
                    f.write(f"[{timestamp}] {indent}{message}\n")
            except Exception as e:
                # Avoid infinite loop if logging fails
                print(f"FATAL ERROR: Failed to write to run log file {self.run_log_file}: {e}. Further file logging disabled.")
                self.run_log_file = None # Disable further logging
    
    def _log_gamestate(self, phase: str, state: PublicGameState):
        """Logs the game state JSON to the consolidated run log file."""
        if self.run_log_file:
            try:
                with open(self.run_log_file, 'a', encoding='utf-8') as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(f"[{timestamp}] === Game State After Phase: {phase} ===\n")
                    state_dict = {
                        "centers": state.centers,
                        "units": state.units,
                        "builds_disbands": state.builds_disbands
                    }
                    # Use dumps for string representation to write
                    state_json_string = json.dumps(state_dict, indent=2)
                    f.write(state_json_string + "\n") # Add newline after JSON
                    f.write(f"[{timestamp}] === End Game State ===\n")
            except Exception as e:
                print(f"FATAL ERROR: Failed to write game state to run log file {self.run_log_file}: {e}. Further file logging disabled.")
                self.run_log_file = None # Disable further logging

    def run_game(self, max_phases: int = -1):
        """
        Runs the main game loop until the game is done or max_phases is reached.

        Args:
            max_phases: Maximum number of game phases to simulate (-1 for unlimited).
        """
        self._log_message("\n=== Framework: Starting Game Run ===")
        phase_count = 0
        while not self.engine_adapter.is_game_done():
            if max_phases != -1 and phase_count >= max_phases:
                self._log_message(f"Framework: Reached max phases ({max_phases}). Stopping.")
                break

            current_phase_name = self.engine_adapter.get_current_phase()
            self._log_message(f"\n--- Framework: Starting Phase {current_phase_name} --- ({phase_count + 1}) ---")

            # 1. Get Public State (will be used in context assembly)
            public_state = self.engine_adapter.get_public_game_state()

            # --- Negotiation Phase ---
            for round_num in range(1, self.num_negotiation_rounds + 1):
                interaction_type = f"NEGOTIATION_ROUND_{round_num}"
                self._log_message(f"  -- Framework: Starting {interaction_type} --")
                # Deliver messages staged from the previous round/phase
                current_inboxes = self._deliver_staged_messages()

                for power_name in self.powers:
                    agent = self.agents[power_name]
                    self._log_message(f"    - Framework: Activating {power_name} ({agent.__class__.__name__})")
                    context = self._assemble_context(
                        power_name,
                        interaction_type,
                        public_state,
                        current_inboxes.get(power_name, [])
                    )
                    try:
                        # Pass run_log_dir to take_turn
                        action_calls = agent.take_turn(context, self.run_log_dir)
                        self._process_tool_calls(power_name, action_calls, interaction_type)
                    except Exception as e:
                        self._log_message(f"ERROR: Agent {power_name} failed during {interaction_type}: {e}")
                        # Decide how to handle agent failure (e.g., skip, submit default actions)
                        self._process_tool_calls(power_name, [ActionToolCall(tool_name="finish_negotiation_round")], interaction_type) # Ensure it finishes

            # --- Order Submission Phase ---
            interaction_type = "ORDER_SUBMISSION"
            self._log_message(f"  -- Framework: Starting {interaction_type} --")
            # Deliver messages staged from the *last* negotiation round
            current_inboxes = self._deliver_staged_messages()
            self._current_orders = {p: [] for p in self.powers} # Reset orders for this phase

            for power_name in self.powers:
                agent = self.agents[power_name]
                self._log_message(f"    - Framework: Activating {power_name} for orders ({agent.__class__.__name__})")
                context = self._assemble_context(
                    power_name,
                    interaction_type,
                    public_state,
                    current_inboxes.get(power_name, [])
                )
                try:
                    # Pass run_log_dir to take_turn
                    action_calls = agent.take_turn(context, self.run_log_dir)
                    self._process_tool_calls(power_name, action_calls, interaction_type)
                except Exception as e:
                     self._log_message(f"ERROR: Agent {power_name} failed during {interaction_type}: {e}")
                     self._process_tool_calls(power_name, [ActionToolCall(tool_name="finish_orders")], interaction_type)

            # --- Adjudication ---
            self._log_message(f"  -- Framework: Submitting orders for {current_phase_name} --")
            # Only submit if orders were actually generated (handle retreats/builds where some powers might have none)
            if any(self._current_orders.values()):
                 # Log submitted orders
                 for power_name, orders in self._current_orders.items():
                     self._log_message(f"    {power_name} Orders: {orders}")
                 for power_name, orders in self._current_orders.items():
                     self.engine_adapter.set_orders(power_name, orders if orders else [])
            else:
                 # Handle phases where no orders are possible/expected (e.g., maybe after a retreat phase with no retreats)
                 # We might still need to submit empty orders for all powers if the engine requires it.
                 # Check diplomacy lib requirements. For now, assume submitting nothing is okay if no agent generated orders.
                 self._log_message(f"  -- Framework: No orders generated by agents for {current_phase_name}. Skipping submission.")
                 # Alternative: Submit empty orders for all
                 # for power_name in self.powers:
                 #     self.engine_adapter.set_orders(power_name, [])

            # Store the state *before* processing the turn for the *next* phase's history
            prev_state = public_state 
            phase_just_completed = current_phase_name

            # Process turn and capture results
            results_this_phase = self.engine_adapter.process_turn()
            self._last_phase_results_by_power = results_this_phase # Keep for context assembly

            # Generate summary *after* processing
            current_state_after_processing = self.engine_adapter.get_public_game_state() # Get state *after* processing for events
            recent_events = self._generate_recent_events(current_state_after_processing, prev_state)
            
            # Call the LLM summarizer (using the static method)
            # Need to get a client instance and the CORRECT model name from one of the LLM agents
            summary_client = None
            summary_model_name = None # Store the actual model name
            for agent in self.agents.values():
                if isinstance(agent, LLMAgent):
                    summary_client = agent.client # Get the configured client
                    summary_model_name = agent.model_name # <<< Use the agent's actual model name
                    break
            
            if summary_client and summary_model_name:
                # Call summarize_history (it already prints/logs internally if needed)
                summary_text = LLMAgent.summarize_history(
                    previous_summary=self._current_history_summary,
                    recent_events=recent_events,
                    phase_completed=phase_just_completed,
                    client=summary_client, # Pass the client
                    model_name=summary_model_name, # <<< Pass the correct model name
                    temperature=0.2, # Use a specific, lower temp for summarization
                    max_tokens=1000   # Use specific max tokens for summarization - Increased from 500
                )
                # Log the generated summary
                self._log_message(f"  -- Framework: History Summary after {phase_just_completed} --\n{summary_text}")
                self._current_history_summary = summary_text # Update stored summary
            else:
                 self._log_message("Framework: Warning - No LLMAgent found or model name missing to perform summarization. Skipping.")
                 # Fallback if no LLM agent exists (e.g., all HoldAgents)
                 self._current_history_summary += f"\n\nDuring {phase_just_completed}: {', '.join(recent_events)}. Order execution varied across powers." 

            # Update previous state for the *next* iteration's history comparison
            self._previous_public_state = current_state_after_processing

            # Log game state AFTER processing the phase
            self._log_gamestate(phase_just_completed, current_state_after_processing)

            phase_count += 1
            time.sleep(0.1) # Small delay for readability

        # --- Game End ---
        self._log_message("\n=== Framework: Game Run Ended ===")
        final_state = self.engine_adapter.get_final_centers()
        if final_state:
            self._log_message(f"Final Center Counts: {final_state}")
            winners = [p for p, centers in final_state.items() if len(centers) >= 18]
            if winners:
                self._log_message(f"Winner(s) (>= 18 centers): {winners}")
            else:
                self._log_message("No solo winner.")
        else:
            self._log_message("Game ended unexpectedly or before completion criteria met.")

        # Option to save the game log
        # self.engine_adapter.save_game("final_game.json")


    def _deliver_staged_messages(self) -> Dict[PowerName, List[Message]]:
        """Moves messages from staging to a dict representing current inboxes and clears staging."""
        delivered_inboxes = self._message_staging
        self._message_staging = {p: [] for p in self.powers} # Clear staging for next round
        # print(f"Framework: Delivering messages: {delivered_inboxes}") # Debug
        return delivered_inboxes

    def _generate_recent_events(self, current_state: PublicGameState, prev_state: Optional[PublicGameState]) -> List[str]:
        """Compares current and previous state to generate simple event descriptions."""
        if not prev_state:
            return ["Game Start"] # No previous state to compare

        events = []
        all_powers = list(current_state.centers.keys()) # Assume centers dict covers all active powers
        
        # Check for gains/losses in supply centers
        for power in all_powers:
            current_centers = set(current_state.centers.get(power, []))
            prev_centers = set(prev_state.centers.get(power, []))
            
            gained = current_centers - prev_centers
            lost = prev_centers - current_centers
            
            for center in gained:
                # Find who lost it (if anyone)
                loser = None
                for other_power, other_prev_centers in prev_state.centers.items():
                    if center in other_prev_centers:
                         loser = other_power
                         break
                if loser:
                    events.append(f"{power} captured {center} from {loser}")
                else:
                    events.append(f"{power} gained neutral center {center}")
            
            # We don't explicitly log losses here, as they are implied by captures.
            # Logging explicit losses could be added if desired.
            # for center in lost:
            #     events.append(f"{power} lost {center}")
        
        # TODO: Add detection for dislodges, builds, disbands based on unit changes / build counts

        if not events:
             return ["No major center changes detected."]
             
        return events

    def _assemble_context(
        self, power_name: PowerName,
        interaction_type: str,
        public_state: PublicGameState,
        current_inbox: List[Message]
    ) -> AgentContextBundle:
        """Creates the context bundle for the agent."""
        # TODO: Implement better history summarization (msg summaries)
        # TODO: Implement more selective memory retrieval

        # Use the centrally managed summary
        history = HistorySummary(summary_text=self._current_history_summary, last_phase_results=self._last_phase_results_by_power)
        
        # Retrieve the agent's full memory for now
        memory_snippet = self._agent_memory.get(power_name, OrderedDict()) 

        phase_info = PhaseInfo(
            phase_name=self.engine_adapter.get_current_phase(),
            interaction_type=interaction_type,
            negotiation_rounds_total=self.num_negotiation_rounds
        )

        # Placeholder for agent-specific instructions - load from config later?
        # Basic examples:
        base_instruction = "Your goal is to win by controlling 18 supply centers. Communicate strategically, form alliances, and betray when necessary."
        instructions_map = {
            "AUSTRIA": "You are Austria. Surrounded and vulnerable. Secure your core territories (Vie, Bud, Tri) and try to ally with Italy or Russia against Turkey.",
            "ENGLAND": "You are England. An island nation. Secure the seas around you, aiming for Scandinavia and potentially France. Watch out for France and Germany.",
            "FRANCE": "You are France. Corner position. Expand into Iberia (Spa, Por) and potentially Belgium/Burgundy. Balance relations with England and Germany.",
            "GERMANY": "You are Germany. Central position. Aim for Scandinavia, Benelux (Bel, Hol), and potentially Warsaw. Be wary of encirclement by England, France, and Russia.",
            "ITALY": "You are Italy. Focused peninsula. Aim for Tunis and potentially Austria or Turkey. The Lepanto opening against Turkey (with Austrian help) is common.",
            "RUSSIA": "You are Russia. Large but slow. Secure your southern centers (Sev, Rum, War) and decide whether to focus north (Scandinavia) or south (Turkey/Austria). Watch out for England in the north.",
            "TURKEY": "You are Turkey. Corner position. Secure the Black Sea and aim for the Balkans (Bul, Gre, Ser, Rum). Often clashes with Austria and Russia."
        }
        instructions = instructions_map.get(power_name, base_instruction) # Fallback to base instruction

        return AgentContextBundle(
            power_name=power_name,
            agent_instructions=instructions,
            current_phase_info=phase_info,
            public_game_state=public_state,
            history_summary=history, # Use updated history object
            communication_inbox=current_inbox,
            private_memory_snippet=memory_snippet # Pass the retrieved memory
        )

    def _process_tool_calls(
        self, power_name: PowerName,
        action_calls: List[ActionToolCall],
        interaction_type: str
    ) -> None:
        """Executes the sequence of actions requested by the agent."""
        self._log_message(f"    - Framework: Processing {len(action_calls)} actions for {power_name}...")
        finished_correctly = False
        expected_finish_tool = "finish_negotiation_round" if interaction_type.startswith("NEGOTIATION") else "finish_orders"

        for call in action_calls:
            # print(f"      Action: {call.tool_name}, Args: {call.arguments}") # Debug
            if call.tool_name == "send_message":
                if not interaction_type.startswith("NEGOTIATION"):
                     self._log_message(f"Warning: {power_name} tried send_message outside negotiation phase. Ignored.")
                     continue
                try:
                    recipient = call.arguments['recipient']
                    msg = Message(
                        sender=power_name,
                        recipient=recipient,
                        message_type=call.arguments['message_type'],
                        content=call.arguments['content'],
                        turn=self.engine_adapter.get_current_phase(),
                        round=int(interaction_type.split('_')[-1]) if interaction_type.startswith("NEGOTIATION") else None
                    )
                    if recipient == 'BROADCAST':
                        for p in self.powers:
                            if p != power_name:
                                self._message_staging[p].append(msg)
                                self._log_message(f"    -> Message Staged: {power_name} -> {p}: {msg.content[:50]}...") # Log message sending
                    elif recipient in self.powers:
                        self._message_staging[recipient].append(msg)
                        self._log_message(f"    -> Message Staged: {power_name} -> {recipient}: {msg.content[:50]}...") # Log message sending
                    else:
                        self._log_message(f"Warning: {power_name} sent message to invalid recipient '{recipient}'. Ignored.")
                except KeyError as e:
                    self._log_message(f"Warning: {power_name} called send_message with missing argument: {e}")
                except Exception as e:
                     self._log_message(f"Error processing send_message for {power_name}: {e}")

            elif call.tool_name == "submit_order":
                 if interaction_type != "ORDER_SUBMISSION":
                     self._log_message(f"Warning: {power_name} tried submit_order outside order phase. Ignored.")
                     continue
                 try:
                     order_str = call.arguments['order_string']
                     # TODO: Add basic validation? Or rely on engine adapter?
                     self._current_orders[power_name].append(order_str)
                     self._log_message(f"    -> Order Queued: {power_name}: {order_str}") # Log order queuing
                 except KeyError:
                     self._log_message(f"Warning: {power_name} called submit_order with missing 'order_string'.")
                 except Exception as e:
                     self._log_message(f"Error processing submit_order for {power_name}: {e}")

            elif call.tool_name == "update_memory":
                try:
                    key = call.arguments['key']
                    value = call.arguments['value']
                    # Get the specific agent's memory (which is an OrderedDict)
                    agent_mem = self._agent_memory.setdefault(power_name, OrderedDict())
                    agent_mem[key] = value # Add/update the item
                    # Enforce memory limit
                    while len(agent_mem) > self.memory_limit:
                        oldest_key, _ = agent_mem.popitem(last=False) # Remove the oldest item
                        self._log_message(f"    -> Memory Pruned ({power_name}): Removed oldest item '{oldest_key}' due to limit ({self.memory_limit}).")
                        
                    self._log_message(f"    -> Memory Updated ({power_name}): {key} = {str(value)[:50]}...")
                except KeyError:
                     self._log_message(f"Warning: {power_name} called update_memory with missing 'key' or 'value'.")
                except Exception as e:
                     self._log_message(f"Error processing update_memory for {power_name}: {e}")

            elif call.tool_name == "log_thought":
                 try:
                     thought = str(call.arguments.get('thought', '(no thought content provided)'))
                     # Log thought to framework log instead of just printing
                     self._log_message(f"    >> Thought ({power_name}): {thought}") 
                 except Exception as e:
                      self._log_message(f"Error processing log_thought for {power_name}: {e}")

            elif call.tool_name == expected_finish_tool:
                finished_correctly = True
                self._log_message(f"    -> Finish tool ({call.tool_name}) called by {power_name}.") # Log finish
                break # Stop processing further actions after finish
            else:
                self._log_message(f"Warning: {power_name} called unexpected tool '{call.tool_name}' during {interaction_type}. Ignored.")

        if not finished_correctly:
            self._log_message(f"Warning: Agent {power_name} did not call {expected_finish_tool} at the end of its turn.")

# Example Usage (for testing later)
# This entire block is redundant because the main() function below does the setup and run.
# if __name__ == '__main__':
#     # Ensure diplomacy engine is accessible via PYTHONPATH
#     # Example: PYTHONPATH=$PYTHONPATH:../diplomacy_engine python -m diplomacy_agent_framework.core.framework
#     # Or run from the workspace root:
#     # PYTHONPATH=./diplomacy_engine python -m diplomacy_agent_framework.core.framework
    
#     # Define which agent class to use for each power
#     # Example: Use LLMAgent for France, HoldAgent for others
#     # agent_mapping = {
#     #     "FRANCE": LLMAgent,
#     #     # Add other powers explicitly if you want them to be HoldAgent or another type
#     #     # If a power is not listed, FrameworkOrchestrator defaults to HoldAgent
#     # }

#     # Use LLMAgent for ALL powers
#     # Get powers from the engine adapter directly
#     temp_adapter = EngineAdapter(map_name='standard') 
#     all_powers_list = temp_adapter.get_all_powers()
#     del temp_adapter # Clean up temporary adapter
#     agent_mapping = {power: LLMAgent for power in all_powers_list}

#     # Instantiate agents with necessary configurations
#     agents_instances = {}
#     default_llm_config = {
#          "llm_model_name": "tgi", 
#          "llm_temperature": 0.1, # Keeping temperature at 0.1 for now
#          "llm_max_tokens": 1024  
#     }
#     # Get all powers first to iterate
#     temp_adapter = EngineAdapter(map_name='standard') 
#     all_powers_list = temp_adapter.get_all_powers()
#     del temp_adapter # Clean up temporary adapter
#     agent_mapping = {power: LLMAgent for power in all_powers_list}
    
#     # Use the already retrieved list of powers
#     for power_name in all_powers_list:
#         agent_class = agent_mapping.get(power_name, HoldAgent) # Fallback just in case
#         if agent_class == LLMAgent:
#             # Pass power_name and config to LLMAgent constructor
#             agents_instances[power_name] = LLMAgent(power_name, **default_llm_config)
#         else:
#             agents_instances[power_name] = agent_class(power_name)
    
#     # Instantiate the framework using the agent *instances*
#     # Ensure orchestrator init knows which agents are which power
#     framework = FrameworkOrchestrator(agent_instances=agents_instances, map_name='standard')
    
#     # Run for a limited number of phases to test LLM interaction
#     framework.run_game(max_phases=3) # Let's stick to 3 phases for now 


# --- Main Execution Block ---
def main():
    # Load environment variables (consider moving if needed elsewhere)
    load_dotenv()

    # --- Create Log Directories ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root_dir = "logs"
    run_log_dir = os.path.join(log_root_dir, f"run_{timestamp}")
    try:
        os.makedirs(run_log_dir, exist_ok=True)
        print(f"Framework: Logging this run to: {run_log_dir}")
    except OSError as e:
        print(f"ERROR: Could not create log directory {run_log_dir}: {e}")
        return # Exit if we can't create log dir

    # --- Agent Configuration ---
    # Define which agent class to use for each power.
    # Using LLMAgent for all powers in this example.

    # ** LLM Configuration **
    # These will be passed to each LLMAgent instance
    llm_model_name = "tgi" # Model identifier for your endpoint - NOTE: Review if this needs update for OpenRouter?
    llm_temperature = 0.1 # For agent decision making
    llm_max_tokens = 8000 # Max tokens for agent decision making - Increased from 4096

    # ** Generalized Agent Instructions **
    general_instructions = {
        "AUSTRIA": "You are Austria. Secure your home centers (Vienna, Budapest, Trieste). Manage relationships with Italy, Russia, and Turkey to navigate the complex Balkan situation. Aim for expansion while maintaining defensive stability.",
        "ENGLAND": "You are England. Leverage your island position to build naval dominance. Secure the British Isles and expand into nearby coastal regions like Scandinavia, France, or Germany. Manage naval competition carefully.",
        "FRANCE": "You are France. As a corner power, your primary expansion paths are towards the Iberian peninsula and the Low Countries (Belgium, Holland). Secure your homeland (Paris, Brest, Marseilles) and manage your relationships with England and Germany carefully to avoid early conflict.", # Updated France
        "GERMANY": "You are Germany. Centrally located, you have multiple expansion options (Scandinavia, Benelux, Russia, Austria). Balance diplomacy and military action carefully to avoid being attacked from multiple sides. Secure your home centers (Berlin, Kiel, Munich).",
        "ITALY": "You are Italy. Secure the Italian peninsula (Rome, Naples, Venice). Focus on Mediterranean expansion, potentially clashing with Austria or Turkey. Consider opportunities in France or the Balkans.",
        "RUSSIA": "You are Russia. With four home centers (Moscow, St. Petersburg, Sevastopol, Warsaw), you have vast territory but are spread thin. Choose your expansion direction (North, South, or towards Central Europe) and manage relationships with neighbors like Turkey, Austria, Germany, and England.",
        "TURKEY": "You are Turkey. Secure your home centers (Constantinople, Ankara, Smyrna) and control the Black Sea. Navigate the Balkan conflicts with Austria and Russia. Consider expansion into the Mediterranean or towards Russia."
    }

    # Create agent instances with specific configurations
    agent_instances = {}
    all_powers = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY'] # Standard powers
    for power in all_powers:
        agent_instances[power] = LLMAgent(
            power_name=power,
            llm_model_name=llm_model_name,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens
        )
        # Assign the general instruction after creation
        # Check if the agent instance has the attribute before assigning
        if hasattr(agent_instances[power], 'agent_instructions'):
            agent_instances[power].agent_instructions = general_instructions.get(power, "You are a Diplomacy power. Play to win.") # Add a default fallback
        else:
             # If the attribute doesn't exist, maybe add it dynamically or log a warning
             # For now, let's add it dynamically (assuming LLMAgent is designed to handle this)
             setattr(agent_instances[power], 'agent_instructions', general_instructions.get(power, "You are a Diplomacy power. Play to win."))
             # Or print a warning: print(f"Warning: Agent for {power} does not have 'agent_instructions' attribute.")

    # --- Orchestrator Setup ---
    orchestrator = FrameworkOrchestrator(
        agent_instances=agent_instances,
        num_negotiation_rounds=2,
        map_name='standard',
        run_log_dir=run_log_dir # <--- Pass the log directory path
    )

    # --- Run Game ---
    orchestrator.run_game()

if __name__ == "__main__":
    main() # Call the main function to start the process 