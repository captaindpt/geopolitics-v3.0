# Diplomacy LLM Agent Framework

This project provides a framework for running Diplomacy agents, including agents powered by Large Language Models (LLMs), using the `diplomacy` library.

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
    ```
    *(Use `.venv\Scripts\activate` on Windows)*

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API Credentials:**
    The framework uses the `python-dotenv` library to load API keys from a `.env` file in the project root. Create a file named `.env` and add your keys. For the configured Hugging Face Inference Endpoint (TGI), you need:
    ```.env
    HF_API_KEY=hf_YOUR_HUGGING_FACE_TOKEN 
    # If using OpenAI directly (not the default setup currently):
    # OPENAI_API_KEY=sk-YOUR_OPENAI_KEY
    # OPENAI_BASE_URL=YOUR_ENDPOINT_URL # If using a custom OpenAI-compatible endpoint
    ```
    Replace `hf_YOUR_HUGGING_FACE_TOKEN` with your actual Hugging Face API token that has access to your inference endpoint. Make sure the `.env` file is included in your `.gitignore` if it isn't already.

## Running the Framework

Ensure your virtual environment is activated. Run the main framework script from the project root directory:

```bash
source .venv/bin/activate && PYTHONPATH=./diplomacy_engine python -m diplomacy_agent_framework.core.framework
```

This command does the following:
*   Activates the virtual environment (`source .venv/bin/activate`).
*   Sets the `PYTHONPATH` so that the Python interpreter can find the `diplomacy_engine` module.
*   Runs the `framework.py` script as a module (`python -m ...`).

The framework will initialize the game, create the LLM agents with generalized goals, and run the game simulation for a few phases (currently configured for 3 phases in `framework.py`), printing logs and agent thoughts to the console.

Log files specifically tracking the context and responses for the FRANCE agent will be created in the root directory with timestamps (e.g., `france_llm_log_YYYYMMDD_HHMMSS.txt`). 