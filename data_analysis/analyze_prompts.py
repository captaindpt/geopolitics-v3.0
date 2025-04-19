import os
import re
import csv
import argparse
from typing import List, Dict, Tuple, Optional

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("WARNING: tiktoken library not found. Token counts will not be calculated.")
    print("Install it using: pip install tiktoken")

# --- Constants ---
DEFAULT_LOGS_DIR = "logs"
# Construct the default output path relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_CSV = os.path.join(SCRIPT_DIR, "prompt_analysis.csv")
# Common encoding for many newer OpenAI models, adjust if needed
DEFAULT_TOKENIZER_ENCODING = "cl100k_base"
# Max characters of prompt to include in CSV snippet
PROMPT_SNIPPET_LENGTH = 100

# --- Regex Patterns ---
# For agent logs (COUNTRY.log)
# Captures: 1=Phase, 2=InteractionType, 3=Prompt Content
AGENT_PROMPT_PATTERN = re.compile(
    r"Phase: (.*?),\s*Interaction: (.*?)\n+"          # Capture Phase and Interaction Type
    r"--- Prompt Sent to LLM ---\n"                  # Start delimiter
    r"(.*?)"                                         # Capture Prompt Content (non-greedy)
    r"\n--- End Prompt ---",                          # End delimiter
    re.DOTALL | re.IGNORECASE                        # DOTALL for multiline prompt, IGNORECASE just in case
)

# For summary logs (summary_prompts.log)
# Captures: 1=Phase, 2=Prompt Content
SUMMARY_PROMPT_PATTERN = re.compile(
    r"===== Summary Prompt for Phase: (.*?) =====\n" # Capture Phase
    r"(.*?)"                                         # Capture Prompt Content (non-greedy)
    r"\n--- End Summary Prompt ---",                 # End delimiter
    re.DOTALL | re.IGNORECASE                       # DOTALL for multiline prompt
)

# --- Helper Functions ---

def get_tokenizer(encoding_name: str = DEFAULT_TOKENIZER_ENCODING):
    """Initializes and returns a tiktoken tokenizer."""
    if not TIKTOKEN_AVAILABLE:
        return None
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        print(f"ERROR: Failed to initialize tiktoken tokenizer '{encoding_name}': {e}")
        return None

def calculate_metrics(prompt_text: str, tokenizer) -> Tuple[int, Optional[int]]:
    """Calculates character length and token count for a prompt."""
    char_length = len(prompt_text)
    token_count = None
    if tokenizer:
        try:
            token_count = len(tokenizer.encode(prompt_text))
        except Exception as e:
            print(f"Warning: Failed to encode prompt snippet for token count: {e}")
    return char_length, token_count

def parse_agent_log(file_path: str, run_id: str, power_name: str, tokenizer) -> List[Dict]:
    """Parses an agent log file and extracts prompt details."""
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Warning: Agent log file not found: {file_path}")
        return []
    except Exception as e:
        print(f"ERROR: Could not read agent log file {file_path}: {e}")
        return []

    matches = AGENT_PROMPT_PATTERN.findall(content)
    for i, match in enumerate(matches):
        phase, interaction_type, prompt_text = match
        prompt_text = prompt_text.strip()
        # Unescape newlines in the prompt text before calculating metrics
        prompt_text_unescaped = prompt_text.replace('\\n', '\n')

        char_length, token_count = calculate_metrics(prompt_text_unescaped, tokenizer)
        results.append({
            "RunID": run_id,
            "Power": power_name,
            "TurnNum": i + 1, # Simple turn counter within the file
            "Phase": phase.strip(),
            "InteractionType": interaction_type.strip(),
            "CharLength": char_length,
            "TokenCount": token_count if token_count is not None else "N/A",
            "PromptSnippet": prompt_text_unescaped[:PROMPT_SNIPPET_LENGTH].replace('\n', ' ') + ("..." if len(prompt_text_unescaped) > PROMPT_SNIPPET_LENGTH else "")
        })
    return results

def parse_summary_log(file_path: str, run_id: str, tokenizer) -> List[Dict]:
    """Parses the summary log file."""
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        # It's okay if the summary log doesn't exist
        return []
    except Exception as e:
        print(f"ERROR: Could not read summary log file {file_path}: {e}")
        return []

    matches = SUMMARY_PROMPT_PATTERN.findall(content)
    for i, match in enumerate(matches):
        phase, prompt_text = match
        prompt_text = prompt_text.strip()
        # Unescape newlines
        prompt_text_unescaped = prompt_text.replace('\\n', '\n')

        char_length, token_count = calculate_metrics(prompt_text_unescaped, tokenizer)
        results.append({
            "RunID": run_id,
            "Power": "Summarizer",
            "TurnNum": i + 1,
            "Phase": phase.strip(),
            "InteractionType": "SUMMARY",
            "CharLength": char_length,
            "TokenCount": token_count if token_count is not None else "N/A",
            "PromptSnippet": prompt_text_unescaped[:PROMPT_SNIPPET_LENGTH].replace('\n', ' ') + ("..." if len(prompt_text_unescaped) > PROMPT_SNIPPET_LENGTH else "")
        })
    return results

def find_run_directories(base_logs_dir: str) -> List[Tuple[str, str]]:
    """Finds all run directories (e.g., 'run_YYYYMMDD_HHMMSS') within the base directory."""
    run_dirs = []
    if not os.path.isdir(base_logs_dir):
        print(f"ERROR: Base logs directory not found: {base_logs_dir}")
        return []
    try:
        for item in os.listdir(base_logs_dir):
            full_path = os.path.join(base_logs_dir, item)
            if os.path.isdir(full_path) and item.startswith("run_"):
                run_dirs.append((item, full_path)) # Store (run_id, full_path)
    except Exception as e:
        print(f"ERROR: Failed to list directories in {base_logs_dir}: {e}")
    return run_dirs

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Analyze LLM prompts from Diplomacy game logs.")
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=DEFAULT_LOGS_DIR,
        help=f"Base directory containing the run log folders (default: {DEFAULT_LOGS_DIR})"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Path to save the analysis results CSV file (default: {DEFAULT_OUTPUT_CSV})"
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default=DEFAULT_TOKENIZER_ENCODING,
        help=f"Tiktoken encoding name to use for token counting (default: {DEFAULT_TOKENIZER_ENCODING})"
    )
    args = parser.parse_args()

    print(f"Starting prompt analysis...")
    print(f"Logs directory: {args.logs_dir}")
    print(f"Output CSV: {args.output_csv}")
    if TIKTOKEN_AVAILABLE:
        print(f"Tokenizer encoding: {args.encoding}")
    else:
        print("Tokenizer: Not available, skipping token counts.")

    tokenizer = get_tokenizer(args.encoding)

    run_directories = find_run_directories(args.logs_dir)
    if not run_directories:
        print("No run directories found. Exiting.")
        return

    print(f"Found {len(run_directories)} run director{'y' if len(run_directories) == 1 else 'ies'}.")

    fieldnames = [
        "RunID", "Power", "TurnNum", "Phase", "InteractionType",
        "CharLength", "TokenCount", "PromptSnippet"
    ]
    prompts_processed_count = 0

    try:
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # Write header immediately
            writer.writeheader()
            print(f"Opened {args.output_csv} for writing.")

            for run_id, run_dir_path in run_directories:
                print(f"\nProcessing run: {run_id} ({run_dir_path})")
                run_prompts_found = 0

                # Process agent logs
                found_agent_logs = False
                try:
                    for filename in os.listdir(run_dir_path):
                        if filename.endswith(".log") and filename not in ["run.log", "summary_prompts.log"]:
                            power_name = filename[:-4] # Remove .log extension
                            file_path = os.path.join(run_dir_path, filename)
                            print(f"  Parsing agent log: {filename}")
                            agent_data = parse_agent_log(file_path, run_id, power_name, tokenizer)
                            if agent_data:
                                # Write rows incrementally
                                writer.writerows(agent_data)
                                prompts_processed_count += len(agent_data)
                                run_prompts_found += len(agent_data)
                            found_agent_logs = True
                except Exception as e:
                    print(f"ERROR: Failed to process agent logs in {run_dir_path}: {e}")

                if not found_agent_logs:
                    print(f"  Warning: No agent logs found in {run_dir_path}")

                # Process summary log
                summary_log_path = os.path.join(run_dir_path, "summary_prompts.log")
                if os.path.exists(summary_log_path):
                    print(f"  Parsing summary log: summary_prompts.log")
                    summary_data = parse_summary_log(summary_log_path, run_id, tokenizer)
                    if summary_data:
                        # Write rows incrementally
                        writer.writerows(summary_data)
                        prompts_processed_count += len(summary_data)
                        run_prompts_found += len(summary_data)
                else:
                    print(f"  Summary log not found (optional): summary_prompts.log")

                print(f"  Finished processing run {run_id}. Found {run_prompts_found} prompts in this run.")

        # --- End of with open block ---

        print("\nAnalysis complete.")
        if prompts_processed_count > 0:
            print(f"Processed a total of {prompts_processed_count} prompts.")
            print(f"Results saved to {args.output_csv}")
        else:
            print("No prompt data was extracted or written.")

    except Exception as e:
        print(f"ERROR: Failed during CSV writing process for {args.output_csv}: {e}")

if __name__ == "__main__":
    main() 