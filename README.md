# Diplomacy AI Agent Framework

## Overview

This project aims to develop sophisticated AI agents capable of playing the complex strategic board game Diplomacy. It utilizes Large Language Models (LLMs) integrated into a custom framework to handle game state, agent interactions, negotiation, and order submission. The framework manages the game loop, interfaces with a Diplomacy engine, and provides context to LLM-based agents.

## Current State 

The framework is under active development. Key components include:

*   **`FrameworkOrchestrator`**: Manages the game loop, phase transitions (negotiation, orders), and agent activation.
*   **`EngineAdapter`**: Interfaces with an underlying Diplomacy game engine (assumed to be in `diplomacy_engine/`).
*   **`LLMAgent`**: An agent implementation that uses an LLM (configured via environment variables to point to an inference endpoint like TGI) to decide actions based on provided context. It uses a tool-calling approach to interact (send messages, submit orders, update memory, log thoughts).
*   **`HoldAgent`**: A simple baseline agent that holds all units.
*   **Context Assembly**: The framework assembles detailed context bundles for agents, including game state, recent summaries, incoming messages, agent memory, and specific instructions.
*   **Debugging & Refinements**: Recent work focused on debugging LLM behavior (specifically for the 'FRANCE' agent), improving context stability, implementing "Think Aloud" logging (`log_thought`), introducing LLM-based state summarization, and generalizing agent goals.

## Key Features

*   Turn-based game loop management compatible with Diplomacy phases.
*   LLM integration for agent decision-making using OpenAI's client library (configurable for custom endpoints).
*   Structured tool-calling interface for agent actions (`send_message`, `submit_order`, `update_memory`, `log_thought`).
*   Distinction between Negotiation rounds and Order Submission phases.
*   Basic persistent agent memory (`_agent_memory`).
*   LLM-powered game state summarization between phases.
*   Timestamped, agent-specific logging for debugging LLM interactions (currently implemented for 'FRANCE').
*   Generalized high-level goals provided to agents.

## Setup

1.  **Clone:** Clone this repository.
2.  **Environment:** Create and activate a Python virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
3.  **Dependencies:** Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Environment Variables:** Create a `.env` file in the project root with the following variables:
    ```dotenv
    OPENAI_API_KEY="YOUR_API_KEY_OR_PLACEHOLDER" # Required by openai client, even if using custom base url
    OPENAI_BASE_URL="YOUR_TGI_OR_INFERENCE_ENDPOINT_URL" # e.g., http://localhost:8080/v1
    ```
    Replace the values with your actual inference endpoint URL and a placeholder API key (the key might not be strictly necessary depending on your endpoint's auth, but the client library often expects it).

## Running the Framework

Execute the main framework module from the project root directory:

```bash
source .venv/bin/activate && PYTHONPATH=./diplomacy_engine python -m diplomacy_agent_framework.core.framework
```

This command:
*   Activates the virtual environment.
*   Adds the `diplomacy_engine` directory to the Python path (assuming the engine code resides there).
*   Runs the `framework.py` module, which initializes the orchestrator and agents, and starts the game simulation (currently set for 3 phases).

Agent interactions, thoughts (if logged), warnings, and phase transitions will be printed to the console. A timestamped log file for France (e.g., `france_llm_log_YYYYMMDD_HHMMSS.txt`) will also be created in the root directory.

## Current Limitations & Next Steps

*   **LLM Reasoning:** Agents currently follow instructions somewhat rigidly and may not exhibit deep strategic reasoning or adaptation, especially in negotiation.
*   **Memory:** Memory usage is basic (simple key-value store). More sophisticated retrieval and summarization techniques are needed.
*   **Negotiation:** Negotiation logic is rudimentary. Agents need better strategies for forming alliances, making proposals, and understanding counter-offers.
*   **Summarization Tuning:** The LLM summarizer prompt might need further tuning for conciseness and relevance.
*   **Error Handling:** Robustness against invalid LLM outputs or engine errors could be improved. 