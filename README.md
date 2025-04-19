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

## Prompt Analysis Tool (`analyze_prompts.py`)

**Motivation:**

LLMs can sometimes lose focus or perform less optimally when presented with excessively long or cluttered prompts. To better understand the context being provided to the agents and the summarizer throughout a game, and to get an intuition on potential context window limitations or prompt efficiency, we developed a dedicated analysis tool.

**Functionality:**

The `analyze_prompts.py` script is specifically designed to parse the log files generated by this framework (`POWER.log` and `summary_prompts.log` within each `logs/run_YYYYMMDD_HHMMSS/` directory). It extracts every prompt sent to the LLM during a simulation run.

For each extracted prompt, it calculates:
*   Character length.
*   Estimated token count (using the `tiktoken` library, requires installation: `pip install tiktoken`).

**Usage:**

1.  Ensure `tiktoken` is installed in your environment (`pip install tiktoken`).
2.  Navigate to the project root directory in your terminal.
3.  Run the script:
    ```bash
    python analyze_prompts.py 
    ```
4.  By default, it reads logs from the `./logs` directory and outputs a CSV file named `prompt_analysis.csv` in the current directory.
5.  You can specify alternative locations:
    *   `python analyze_prompts.py --logs-dir /path/to/other/logs`
    *   `python analyze_prompts.py --output-csv path/to/my_analysis.csv`

**Output:**

The generated CSV file (`prompt_analysis.csv`) contains columns such as `RunID`, `Power`, `TurnNum`, `Phase`, `InteractionType`, `CharLength`, `TokenCount`, and a `PromptSnippet`. This structured data allows for quantitative analysis of prompt sizes across different game stages, agents, and simulation runs, helping to identify potential areas where prompts might become overly large or inefficient.

## Running the Framework

Ensure your virtual environment is activated and your `.env` file is configured with your `OPENROUTER_API_KEY`. Run the main framework script from the project root directory:

```bash
PYTHONPATH=./diplomacy_engine python -m diplomacy_agent_framework.core.framework
```

This command sets the `PYTHONPATH` to include the `diplomacy_engine` module (required for imports) and runs the `framework.py` script. The simulation will run until the game completes or an error occurs, logging detailed information to the console and the corresponding `logs/run_YYYYMMDD_HHMMSS/` directory.

## Alternative: Running via Jupyter Notebook (`diplomacy2.md`)

As an alternative to running the framework script directly, you can use the provided Jupyter Notebook (`diplomacy2.ipynb`). This notebook contains the entire framework code and allows for interactive execution and experimentation.

**Setup using Conda and `environment.yml`:**

1.  **Install Conda:** If you don't have Conda installed, follow the installation instructions for [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or [Anaconda](https://www.anaconda.com/products/distribution).

2.  **Create Conda Environment:** Navigate to the project root directory in your terminal and create the environment using the provided `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```
    This command creates a new Conda environment named `diplomacy_env` (as defined in the `environment.yml` file) with all the necessary dependencies.

3.  **Activate the Environment:**
    ```bash
    conda activate diplomacy_env
    ```

4.  **Install IPython Kernel:** To make this environment available in Jupyter, install an IPython kernel specific to it:
    ```bash
    python -m ipykernel install --user --name diplomacy_env --display-name "Python (diplomacy_env)"
    ```

5.  **Launch Jupyter:** Start Jupyter Lab or Jupyter Notebook:
    ```bash
    jupyter lab 
    # or
    # jupyter notebook
    ```

6.  **Open and Run Notebook:**
    *   In the Jupyter interface, navigate to and open the `diplomacy2.md` file.
    *   Ensure you select the `Python (diplomacy_env)` kernel when prompted or via the Kernel menu.
    *   **Crucially**, update the `OPENROUTER_API_KEY` variable in the first code cell with your actual key.
    *   Run the cells sequentially to execute the simulation setup and run the game.

This notebook approach is particularly useful for debugging, modifying agent behavior, or stepping through the game logic. Remember to keep the API key secure and configure it within the notebook before running. 