import os
import re
import json
import argparse
import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, OrderedDict
import time

try:
    from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError, APIStatusError
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: openai library not found. LLM analysis will not be possible.")
    print("Install it using: pip install openai")
    OPENAI_AVAILABLE = False

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    print("Warning: python-dotenv not found. Cannot load API key from .env file.")
    print("Install it using: pip install python-dotenv")
    DOTENV_AVAILABLE = False

# --- Constants ---
DEFAULT_CHUNK_SIZE = 10
DEFAULT_OUTPUT_FILE = "llm_agent_analysis.json"
DEFAULT_LLM_MODEL = "deepseek/deepseek-r1-zero:free" # Or your preferred model
DEFAULT_LLM_TEMPERATURE = 0.5
DEFAULT_LLM_MAX_TOKENS_STAGE1 = 1024
DEFAULT_LLM_MAX_TOKENS_STAGE2 = 2048
API_KEY_ENV_VAR = "OPENROUTER_API_KEY" # Matches framework
API_BASE_URL_ENV_VAR = "OPENROUTER_API_BASE" # Optional: if using OpenRouter non-default base
DEFAULT_API_BASE_URL = "https://openrouter.ai/api/v1"

# --- Regex Patterns for run.log (Corrected and Separated) ---
# Base pattern for timestamp prefix - EXTREMELY Simplified
TIMESTAMP_RE_STR = 'r\"^\\[\"' # Matches a literal [ at the start of the line

# Define pattern strings first, ensure raw strings (r'...') are used consistently where possible
PHASE_START_STR = TIMESTAMP_RE_STR + r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\]\s+--- Framework: Starting Phase (\S+) ---' # Add rest of pattern
AGENT_ACTIVATION_STR = TIMESTAMP_RE_STR + r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\]\s+\s+- Framework: Activating (\w+)(?: for orders)? \(LLMAgent\)'
THOUGHT_HEADER_STR = TIMESTAMP_RE_STR + r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\]\s+\s+Thought:'
THOUGHT_CONTENT_STR = TIMESTAMP_RE_STR + r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\]\s+\s+\| (.*)'
SENT_MESSAGE_STR = TIMESTAMP_RE_STR + r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\]\s+\s+DEBUG: Processing send_message\. Recipient=\'(\w+)\'.*?, Content=\'(.*?)\'\s+\(Type: <class \'str\'>\)'
ORDERS_STR = TIMESTAMP_RE_STR + r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\]\s+\s+(\w+) Orders: (\[.*?\])'
GAME_STATE_START_STR = TIMESTAMP_RE_STR + r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\]\s+=== Game State After Phase: (\S+) ==='
GAME_STATE_END_STR = TIMESTAMP_RE_STR + r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\]\s+=== End Game State ==='

# Compile patterns
# Add re.MULTILINE if matching start of line (^) within the full log content
PHASE_START_PATTERN = re.compile(PHASE_START_STR, re.MULTILINE)
AGENT_ACTIVATION_PATTERN = re.compile(AGENT_ACTIVATION_STR, re.MULTILINE)
THOUGHT_HEADER_PATTERN = re.compile(THOUGHT_HEADER_STR, re.MULTILINE)
THOUGHT_CONTENT_PATTERN = re.compile(THOUGHT_CONTENT_STR, re.MULTILINE)
SENT_MESSAGE_PATTERN = re.compile(SENT_MESSAGE_STR, re.MULTILINE)
ORDERS_PATTERN = re.compile(ORDERS_STR, re.MULTILINE)
GAME_STATE_START_PATTERN = re.compile(GAME_STATE_START_STR, re.MULTILINE)
GAME_STATE_END_PATTERN = re.compile(GAME_STATE_END_STR, re.MULTILINE)

# --- Helper Functions ---

def load_api_key() -> Optional[str]:
    """Loads the API key from .env file or environment variables."""
    if DOTENV_AVAILABLE:
        load_dotenv() # Load variables from .env file

    api_key = os.getenv(API_KEY_ENV_VAR)
    if not api_key:
        print(f"ERROR: API key not found in environment variable '{API_KEY_ENV_VAR}' or .env file.")
        return None
    if api_key == "sk-or-v1-YOUR_KEY_HERE" or "YOUR_OPENROUTER_KEY" in api_key:
        print(f"ERROR: Placeholder API key detected in '{API_KEY_ENV_VAR}'. Please set your actual key.")
        return None
    return api_key

def get_openai_client(api_key: str) -> Optional[OpenAI]:
    """Initializes the OpenAI client, configured for OpenRouter."""
    if not OPENAI_AVAILABLE:
        return None

    base_url = os.getenv(API_BASE_URL_ENV_VAR, DEFAULT_API_BASE_URL)
    print(f"Using API Base URL: {base_url}")

    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=120.0 # Longer timeout for potentially long analysis
        )
        # Try a simple request to verify the key/connection (optional but recommended)
        # client.models.list() # This might incur cost/rate limit
        print("OpenAI client initialized.")
        return client
    except Exception as e:
        print(f"ERROR: Failed to initialize OpenAI client: {e}")
        return None

def safe_json_loads(json_string: str, context: str = "") -> Optional[Dict]:
    """Attempts to load JSON, returning None on failure."""
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"[WARN] Failed to decode JSON {context}: {e}. Content: {json_string[:200]}...")
        return None
    except Exception as e:
        print(f"[WARN] Unexpected error loading JSON {context}: {e}. Content: {json_string[:200]}...")
        return None

def safe_eval_list_str(list_str: str, context: str = "") -> List[str]:
    """Attempts to evaluate a string representation of a list (e.g., orders). Use with caution."""
    list_str = list_str.strip()
    if not (list_str.startswith('[') and list_str.endswith(']')):
        print(f"[WARN] Invalid list format {context}: {list_str}")
        return []
    try:
        evaluated_list = json.loads(list_str.replace("'", '"')) # Replace single quotes for JSON compatibility
        if isinstance(evaluated_list, list):
            return [str(item) for item in evaluated_list]
        else:
            print(f"[WARN] Evaluated list string is not a list {context}: {list_str}")
            return []
    except Exception as e:
        print(f"[WARN] Failed to evaluate list string {context}: {e}. String: {list_str}")
        return []

# --- Main Parsing Function ---

def parse_run_log(run_log_path: str) -> Dict[str, Any]:
    """Parses the run.log file line-by-line to extract structured phase data."""
    print(f"DEBUG: Starting parse_run_log for {run_log_path}...")

    try:
        with open(run_log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"DEBUG: Read {len(lines)} lines from log file.")
    except Exception as e:
        print(f"ERROR: Could not read run log file {run_log_path}: {e}")
        return {}

    # Store states found, mapping phase name to state dict
    game_states_after_phase = {}
    in_game_state_block = False
    current_state_phase = None
    current_state_lines = []
    print("DEBUG: Pre-parsing for Game State blocks...")
    for i, line in enumerate(lines): # Use enumerate for line numbers
        gs_start_match = GAME_STATE_START_PATTERN.match(line) 
        gs_end_match = GAME_STATE_END_PATTERN.match(line)

        if gs_start_match:
            # print(f"DEBUG Line {i+1}: Matched Game State Start: {gs_start_match.group(1)}")
            in_game_state_block = True
            current_state_phase = gs_start_match.group(1)
            current_state_lines = []
            continue
        
        if gs_end_match and in_game_state_block:
            # print(f"DEBUG Line {i+1}: Matched Game State End for {current_state_phase}")
            state_json_str = '\n'.join(current_state_lines).strip()
            game_states_after_phase[current_state_phase] = safe_json_loads(state_json_str, f"after phase {current_state_phase}")
            in_game_state_block = False
            current_state_phase = None
            current_state_lines = []
            continue
            
        if in_game_state_block:
            current_state_lines.append(line.strip())
    print(f"DEBUG: Finished pre-parsing. Found {len(game_states_after_phase)} game state blocks.")

    # --- Now process events line-by-line --- 
    ordered_phases = OrderedDict()
    current_phase = None
    current_power = None
    current_phase_data = None
    last_game_state = None
    thought_buffer = []
    in_thought_block = False
    print("DEBUG: Starting main event parsing loop...")

    for i, line in enumerate(lines):
        # --- Add specific pattern matching debug --- 
        if i < 50: # Only print for the first 50 lines to avoid excessive output
            phase_match_debug = PHASE_START_PATTERN.match(line)
            print(f"DEBUG Line {i+1}: Checking line: '{line[:100]}...'. Phase Match? {bool(phase_match_debug)}")
            if phase_match_debug:
                 print(f"    -> Matched Phase: {phase_match_debug.group(1)}")
        # -------------------------------------------
        
        # Match patterns against the raw line
        phase_match = PHASE_START_PATTERN.match(line)
        activation_match = AGENT_ACTIVATION_PATTERN.match(line)
        thought_header_match = THOUGHT_HEADER_PATTERN.match(line)
        thought_content_match = THOUGHT_CONTENT_PATTERN.match(line)
        message_match = SENT_MESSAGE_PATTERN.match(line)
        orders_match = ORDERS_PATTERN.match(line)

        # --- State Management --- 
        if phase_match:
            # Finish previous thought block if any
            if in_thought_block and current_power and current_phase_data:
                if current_power in current_phase_data['powers']:
                    thought_text = '\n'.join(thought_buffer).strip()
                    current_phase_data['powers'][current_power]['thought'] = thought_text
                    print(f"DEBUG Line {i+1}: Finished previous thought block for {current_power} before phase change. Length: {len(thought_text)}")
            thought_buffer = []
            in_thought_block = False
            
            current_phase = phase_match.group(1)
            current_power = None # Reset current power for the new phase
            print(f"DEBUG Line {i+1}: Matched PHASE_START: {current_phase}")
            ordered_phases[current_phase] = {
                'game_state_before': last_game_state,
                'powers': defaultdict(lambda: {'thought': None, 'messages_sent': [], 'orders': []}),
                'game_state_after': game_states_after_phase.get(current_phase)
            }
            current_phase_data = ordered_phases[current_phase]
            last_game_state = current_phase_data['game_state_after'] # Update for the *next* phase
            continue
            
        if not current_phase or not current_phase_data:
            # print(f"DEBUG Line {i+1}: Skipping line (before first phase)")
            continue # Skip lines before first phase

        # --- Event Parsing within a Phase --- 
        if activation_match:
            # Finish previous thought block if switching agents
            if in_thought_block and current_power and current_phase_data:
                 if current_power in current_phase_data['powers']:
                      thought_text = '\n'.join(thought_buffer).strip()
                      current_phase_data['powers'][current_power]['thought'] = thought_text
                      print(f"DEBUG Line {i+1}: Finished previous thought block for {current_power} before agent activation. Length: {len(thought_text)}")
            thought_buffer = []
            in_thought_block = False
            current_power = activation_match.group(1)
            print(f"DEBUG Line {i+1}: Matched AGENT_ACTIVATION: {current_power} in phase {current_phase}")
            # Ensure power entry exists
            if current_power not in current_phase_data['powers']:
                 current_phase_data['powers'][current_power] = {'thought': None, 'messages_sent': [], 'orders': []}
            continue
            
        # --- Handle Thought Blocks --- 
        if thought_header_match and current_power:
            # Finish previous thought block if any (e.g., consecutive thoughts? unlikely but possible)
            if in_thought_block and current_power and current_phase_data:
                 if current_power in current_phase_data['powers']:
                      thought_text = '\n'.join(thought_buffer).strip()
                      current_phase_data['powers'][current_power]['thought'] = thought_text
                      print(f"DEBUG Line {i+1}: Finished previous thought block for {current_power} before new thought header. Length: {len(thought_text)}")
            print(f"DEBUG Line {i+1}: Matched THOUGHT_HEADER for {current_power}")
            in_thought_block = True
            thought_buffer = [] # Reset buffer for new thought
            continue
        
        if thought_content_match and in_thought_block and current_power:
            # print(f"DEBUG Line {i+1}: Matched THOUGHT_CONTENT: {thought_content_match.group(1)[:50]}...")
            thought_buffer.append(thought_content_match.group(1)) # Append captured content
            continue
            
        # If we encounter something else that isn't thought content, the block ended
        if in_thought_block: 
            if current_power and current_phase_data:
                 if current_power in current_phase_data['powers']:
                      thought_text = '\n'.join(thought_buffer).strip()
                      current_phase_data['powers'][current_power]['thought'] = thought_text
                      print(f"DEBUG Line {i+1}: Ended thought block for {current_power} due to non-content line. Length: {len(thought_text)}")
            thought_buffer = []
            in_thought_block = False
            # Process the current line normally now

        # --- Handle Other Events (only if NOT processing thought content) --- 
        if message_match and current_power:
            recipient = message_match.group(1)
            content = message_match.group(2)
            print(f"DEBUG Line {i+1}: Matched SENT_MESSAGE from {current_power} to {recipient}")
            current_phase_data['powers'][current_power]['messages_sent'].append({
                'recipient': recipient,
                'content': content
            })
            continue

        if orders_match:
            power_from_log = orders_match.group(1)
            orders_str = orders_match.group(2)
            print(f"DEBUG Line {i+1}: Matched ORDERS for {power_from_log}")
            # Associate with the power logged on the line
            if power_from_log in current_phase_data['powers']:
                orders_list = safe_eval_list_str(orders_str, f"for {power_from_log} in {current_phase}")
                current_phase_data['powers'][power_from_log]['orders'].extend(orders_list)
            else:
                 # Power might not have been activated if it had no actions other than orders submission
                 # Ensure entry exists before adding orders
                 if power_from_log not in current_phase_data['powers']:
                      current_phase_data['powers'][power_from_log] = {'thought': None, 'messages_sent': [], 'orders': []}
                 orders_list = safe_eval_list_str(orders_str, f"for {power_from_log} in {current_phase}")
                 current_phase_data['powers'][power_from_log]['orders'].extend(orders_list)
            continue
        
        # Add a print for unmatched lines if needed (can be very verbose)
        # else:
        #     if line.strip(): # Avoid printing for empty lines
        #         print(f"DEBUG Line {i+1}: Unmatched line: {line[:100]}")

    # Final check for any trailing thought block
    if in_thought_block and current_power and current_phase_data:
        if current_power in current_phase_data['powers']:
            thought_text = '\n'.join(thought_buffer).strip()
            current_phase_data['powers'][current_power]['thought'] = thought_text
            print(f"DEBUG End: Finished final thought block for {current_power}. Length: {len(thought_text)}")

    # Debug: Print a summary of what was parsed
    print(f"DEBUG: Parsing finished. Found {len(ordered_phases)} phases.")
    for phase, data in ordered_phases.items():
        powers_with_data = list(data.get('powers', {}).keys())
        print(f"  - Phase {phase}: Found data for powers: {powers_with_data}")
        # Optionally print more detail, e.g., if thoughts/orders were found
        # for p, p_data in data.get('powers', {}).items():
        #    print(f"    - {p}: Thought? {p_data.get('thought') is not None}, Orders? {bool(p_data.get('orders'))}")
            
    return {"phases": ordered_phases}

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Analyze Diplomacy agent behavior using LLM.")
    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to the specific run directory (e.g., logs/run_YYYYMMDD_HHMMSS)"
    )
    parser.add_argument(
        "--output-file", "-o",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Path to save the analysis results JSON file (default: {DEFAULT_OUTPUT_FILE})"
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Number of phases to analyze in each Stage 1 LLM call (default: {DEFAULT_CHUNK_SIZE})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=f"LLM model name to use for analysis (default: {DEFAULT_LLM_MODEL})"
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=DEFAULT_LLM_TEMPERATURE,
        help=f"LLM temperature for analysis (default: {DEFAULT_LLM_TEMPERATURE})"
    )
    parser.add_argument(
        "--max-tokens-stage1",
        type=int,
        default=DEFAULT_LLM_MAX_TOKENS_STAGE1,
        help=f"Max tokens for Stage 1 LLM response (default: {DEFAULT_LLM_MAX_TOKENS_STAGE1})"
    )
    parser.add_argument(
        "--max-tokens-stage2",
        type=int,
        default=DEFAULT_LLM_MAX_TOKENS_STAGE2,
        help=f"Max tokens for Stage 2 LLM response (default: {DEFAULT_LLM_MAX_TOKENS_STAGE2})"
    )

    args = parser.parse_args()

    print("--- LLM Agent Analysis Script ---")

    # Validate run directory
    run_log_path = os.path.join(args.run_dir, "run.log")
    if not os.path.isdir(args.run_dir) or not os.path.isfile(run_log_path):
        print(f"ERROR: Invalid run directory specified. '{args.run_dir}' is not a directory or does not contain run.log.")
        return

    print(f"Analyzing run: {args.run_dir}")
    print(f"Output file: {args.output_file}")
    print(f"Phase chunk size: {args.chunk_size}")
    print(f"LLM Model: {args.model}")

    # Load API Key and initialize client
    api_key = load_api_key()
    if not api_key:
        return
    client = get_openai_client(api_key)
    if not client:
        print("ERROR: Failed to initialize LLM client. Cannot proceed with analysis.")
        return

    # 1. Parse the run log
    parsed_data = parse_run_log(run_log_path)
    if not parsed_data or 'phases' not in parsed_data or not parsed_data['phases']:
        print("ERROR: Failed to parse run log or no phase data found.")
        return

    # Get phases in order
    ordered_phase_names = list(parsed_data['phases'].keys())
    print(f"Log parsing complete (found {len(ordered_phase_names)} phases). First: {ordered_phase_names[0]}, Last: {ordered_phase_names[-1]}")

    # 2. Perform Analysis
    final_analysis = defaultdict(lambda: {"stage1": [], "stage2": ""})
    powers_in_game = set()
    # Determine powers involved from parsed data
    for phase_name, phase_content in parsed_data['phases'].items():
        if 'powers' in phase_content:
            powers_in_game.update(phase_content['powers'].keys())

    if not powers_in_game:
        print("ERROR: No power data found in parsed phases.")
        return
        
    print(f"Identified powers: {sorted(list(powers_in_game))}")

    for power in sorted(list(powers_in_game)):
        print(f"\nProcessing analysis for: {power}")
        power_phase_chunks = []
        current_chunk = []
        # Extract data relevant to this power chronologically
        for phase_name in ordered_phase_names:
            phase_detail = parsed_data['phases'].get(phase_name, {})
            power_actions = phase_detail.get('powers', {}).get(power)
            if power_actions: # If the power was active/logged in this phase
                # Construct the data packet for this phase for this power
                chunk_item = {
                    "name": phase_name,
                    "thought": power_actions.get('thought'),
                    "orders": power_actions.get('orders', []),
                    "messages_sent": power_actions.get('messages_sent', []),
                    "game_state_before": phase_detail.get('game_state_before'),
                    "game_state_after": phase_detail.get('game_state_after')
                }
                current_chunk.append(chunk_item)
                
                if len(current_chunk) >= args.chunk_size:
                    power_phase_chunks.append(list(current_chunk)) # Add copy of chunk
                    current_chunk = [] # Reset for next chunk

        # Add any remaining items as the last chunk
        if current_chunk:
            power_phase_chunks.append(list(current_chunk))
           
        if not power_phase_chunks:
            print(f"  No phase data found for {power}. Skipping analysis.")
            continue
           
        print(f"  Split {power}'s activity into {len(power_phase_chunks)} chunk(s) for Stage 1 analysis.")

        # Run Stage 1 for all chunks
        stage1_summaries = []
        for chunk_idx, chunk in enumerate(power_phase_chunks):
            print(f"   - Analyzing chunk {chunk_idx + 1}/{len(power_phase_chunks)}")
            stage1_result = run_llm_analysis_stage1(
                client, args.model, args.temp, args.max_tokens_stage1,
                power, chunk
            )
            stage1_summaries.append(stage1_result)
            # Optional: Add a small delay between API calls to avoid rate limits
            time.sleep(1)
           
        final_analysis[power]["stage1"] = stage1_summaries # Store all stage 1 results

        # Run Stage 2 if Stage 1 summaries exist
        if final_analysis[power]["stage1"]:
            stage2_result = run_llm_analysis_stage2(
                client, args.model, args.temp, args.max_tokens_stage2,
                power, final_analysis[power]["stage1"]
            )
            final_analysis[power]["stage2"] = stage2_result
        else:
            print(f"  Skipping Stage 2 for {power} (no Stage 1 results).")

    # 3. Save the results
    print(f"\nSaving analysis results to {args.output_file}...")
    try:
        # Use a custom encoder if needed for complex objects, but should be basic types here
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(final_analysis, f, indent=2)
        print("Analysis complete. Results saved.")
    except TypeError as e:
        print(f"ERROR: Failed to serialize results to JSON: {e}")
        print("Attempting to save with non-serializable data removed...")
        # Basic fallback: convert everything to string
        try:
            fallback_analysis = json.loads(json.dumps(final_analysis, default=str))
            with open(args.output_file + ".fallback.json", 'w', encoding='utf-8') as f:
                json.dump(fallback_analysis, f, indent=2)
            print(f"Fallback results saved to {args.output_file}.fallback.json")
        except Exception as fallback_e:
            print(f"ERROR: Fallback JSON saving also failed: {fallback_e}")
    except Exception as e:
        print(f"ERROR: Failed to write results to {args.output_file}: {e}")

if __name__ == "__main__":
    main() 