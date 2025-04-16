# Diplomacy LLM Agent Framework

This project provides a framework for running Diplomacy games featuring agents powered by Large Language Models (LLMs), using the `diplomacy` library as the core game engine.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd geopolitics-v3.0 
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate 
    # Use `.venv\Scripts\activate` on Windows CMD
    # Use `.venv/bin/Activate.ps1` on Windows PowerShell
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # Ensure 'coloredlogs' is included if not already present in requirements.
    ```

4.  **Set up API Credentials:**
    The framework uses `python-dotenv` to load API keys from a `.env` file in the project root. Create a file named `.env` and add your OpenRouter API key:
    ```.env
    # Required for LLM Agent interaction via OpenRouter
    OPENROUTER_API_KEY=sk-or-v1-YOUR_OPENROUTER_KEY 
    ```
    Replace `sk-or-v1-YOUR_OPENROUTER_KEY` with your actual key. Ensure the `.env` file is added to your `.gitignore`.

## Framework Architecture

The simulation is managed by the `FrameworkOrchestrator` class (`diplomacy_agent_framework/core/framework.py`). Its key responsibilities include:

1.  **Game Initialization:** Uses `EngineAdapter` to interact with the underlying `diplomacy` game engine, setting up the map and powers.
2.  **Agent Management:** Instantiates and manages the agent objects for each power (e.g., `LLMAgent` instances).
3.  **Game Loop:** Executes the Diplomacy game turn structure:
    *   **Negotiation Phases:** Runs the configured number of negotiation rounds. For each round:
        *   Delivers messages staged from the previous round.
        *   Assembles context (`AgentContextBundle`) for each agent, including game state, received messages, memory snippets, and phase-specific instructions.
        *   Calls the agent's `take_turn` method.
        *   Processes the returned actions (`ActionToolCall` list), staging messages and updating memory via `_process_tool_calls`.
    *   **Order Submission Phase:**
        *   Delivers messages from the final negotiation round.
        *   Assembles context for order submission.
        *   Calls the agent's `take_turn` method.
        *   Processes actions, queuing orders via `_process_tool_calls`.
    *   **Adjudication:** Submits the queued orders to the `EngineAdapter` and processes the game turn.
    *   **Summarization:** Calls the LLM (via `LLMAgent.summarize_history`) to generate a narrative summary of the completed phase.
    *   **State Update:** Retrieves and logs the new game state.
4.  **State Management:** Tracks message staging (`_message_staging`), order queuing (`_current_orders`), agent memory (`_agent_memory`), and game history summaries (`_current_history_summary`).

## LLM Agent Details (`LLMAgent`)

The `LLMAgent` (`diplomacy_agent_framework/agents/llm_agent.py`) uses an LLM (configured for OpenRouter) to make decisions.

*   **Interaction Model:** The agent interacts with the framework by generating a JSON list of `ActionToolCall` objects. This structured output defines the specific actions the agent wants to take (e.g., sending messages, submitting orders, updating memory). Available tools include:
    *   `log_thought`: Records the agent's reasoning process for the turn.
    *   `send_message`: Sends messages to other powers during negotiation rounds.
    *   `update_memory`: Stores simple key-value information for later retrieval.
    *   `submit_order`: Queues specific orders during the order submission phase.
    *   `finish_negotiation_round` / `finish_orders`: Signals the end of the agent's turn for the current interaction step.
*   **Reasoning (`log_thought`):** The prompt explicitly guides the LLM to perform step-by-step reasoning *first* and include this detailed reasoning within the `thought` argument of the initial `log_thought` tool call in its JSON response. This is the primary mechanism for capturing and logging the agent's decision-making process.
*   **Context Management:** The `_format_prompt` method dynamically builds the prompt based on the current game phase (`interaction_type`). It provides tailored context, including:
    *   Current public game state (units, centers).
    *   Relevant messages received in the current step.
    *   A snippet of the agent's private memory.
    *   Phase-specific instructions and the set of allowed tools for the JSON response.
*   **Memory (`update_memory`):** Agents can persist simple information (like proposed deals or observations about other players) across turns using the `update_memory` tool. The `FrameworkOrchestrator` stores this information per-agent and provides it back in subsequent turns via the `AgentContextBundle`.
*   **Output Parsing (`_parse_llm_response`):** This method processes the raw text response from the LLM API. It includes logic to:
    *   Clean potential extraneous formatting sometimes added by models (e.g., `\boxed{...}` wrappers, markdown code fences like ` ```json`).
    *   Attempt to parse the cleaned string as a JSON list of `ActionToolCall` objects.
    *   Handle potential JSON decoding errors gracefully.
*   **Robustness:**
    *   **Retry Mechanism:** The `take_turn` method includes a loop to automatically retry the LLM API call (up to 3 times) if an error occurs or if the API returns an empty response, improving resilience against transient network or endpoint issues.
    *   **Increased Tokens:** The `llm_max_tokens` parameter (configured in `framework.py`) has been increased (e.g., to 8000) to reduce the chance of the LLM's output being truncated, especially when detailed reasoning is included in `log_thought`.

## Logging System

The framework implements a comprehensive logging system for each run:

1.  **Run Directory:** A unique directory is created for each simulation run inside the `logs/` folder, named using a timestamp (e.g., `logs/run_YYYYMMDD_HHMMSS/`).
2.  **Consolidated Run Log (`run.log`):** Located inside the run directory, this file contains a chronological record of the framework's execution, mirroring the console output but often with more detail. It includes:
    *   Framework status messages (phase starts, agent activations).
    *   Agent thoughts (from the `log_thought` tool).
    *   Queued orders and submitted messages.
    *   Phase results (adjudication outcomes).
    *   JSON dumps of the public game state after each phase completes.
    *   Generated history summaries.
3.  **Agent-Specific Logs (`POWER.log`):** For each agent (e.g., `FRANCE.log`, `GERMANY.log`), a separate log file is created in the run directory. This file contains:
    *   The exact context prompt sent to the LLM for that agent's turn.
    *   The full, raw response received from the LLM API before parsing.
    *   Crucial for debugging the behavior and responses of individual LLM agents.

## Running the Framework

Ensure your virtual environment is activated and your `.env` file is configured with your `OPENROUTER_API_KEY`. Run the main framework script from the project root directory:

```bash
PYTHONPATH=./diplomacy_engine python -m diplomacy_agent_framework.core.framework
```

This command sets the `PYTHONPATH` to include the `diplomacy_engine` module (required for imports) and runs the `framework.py` script. The simulation will run until the game completes or an error occurs, logging detailed information to the console and the corresponding `logs/run_YYYYMMDD_HHMMSS/` directory. 