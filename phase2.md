# Phase 2: Detailed Agent Interaction Model (Sequential Execution)

## 1. Goal

To design and document a detailed operational model for the Diplomacy AI Agent Framework, specifically addressing the constraint of running all agents sequentially within a single computing instance. This model must allow agents to:

*   Perceive the public game state.
*   Engage in multi-round negotiation with other agents.
*   Maintain private state and reasoning.
*   Make final decisions and submit orders to the Diplomacy engine.

## 2. Core Problem & Approach

The primary challenge is simulating the parallel thinking and negotiation inherent in Diplomacy using a strictly sequential execution flow. Agents cannot truly "wait" for replies in real-time. 

The approach is to structure each game phase (e.g., Spring Movement, Fall Retreats, Fall Movement, Winter Adjustments) into distinct sub-phases managed by the Framework/Orchestrator:

1.  **State Update & Context Assembly:** Disseminate the latest public game state and provide necessary context to the agent.
2.  **Negotiation:** Allow structured rounds of communication between agents, driven by agent tool usage.
3.  **Order Submission:** Collect final, binding orders from agents, driven by agent tool usage.
4.  **Adjudication:** Submit orders to the engine and advance the game.

This ensures that each agent gets dedicated processing time for communication strategy and final decision-making, fitting within the sequential constraint, and clarifies the agent-framework interface through defined inputs (context) and outputs (tool calls).

## 3. Detailed Turn Structure (Per Game Phase)

The Framework orchestrates the following sequence for *each* game phase provided by the Diplomacy engine (e.g., S1901M, F1901M, W1901A, S1902M...):

### 3.1. Phase Start, State Update & Context Assembly

1.  **Framework:** Retrieves the current, adjudicated game state from the Diplomacy Engine (`diplomacy.Game` object). This includes unit positions, supply center ownership, current phase/year, etc.
2.  **Framework:** For the agent whose turn it is, assembles the **Agent Context Bundle** (see Section 4).
3.  **Framework:** Calls the primary agent execution method (e.g., `agent.take_turn(context_bundle)`), indicating the current interaction phase (e.g., `NEGOTIATION_ROUND_1`, `ORDER_SUBMISSION`).

### 3.2. Negotiation Phase (Iterative Rounds)

This phase allows agents to exchange messages before committing to orders. It consists of a fixed number of rounds (e.g., **2 rounds** - configurable parameter).

**For each Negotiation Round (e.g., Round 1, Round 2):**

1.  **Framework:** Iterates through each active agent sequentially (e.g., Austria -> England -> France...). Let the current agent be `agent_i`.
2.  **Framework:** Assembles the Context Bundle for `agent_i`, including messages received *before the start of this round*.
3.  **Framework:** Calls `agent_i.take_turn(context_bundle, phase='NEGOTIATION_ROUND_N')`.
4.  **Agent Logic:** Inside `take_turn`, the agent:
    *   Parses the `context_bundle`.
    *   Executes its internal reasoning (potentially using `private_memory`).
    *   Outputs a sequence of **Action Tool Calls** (see Section 5). Key tools for this phase are `send_message` and `update_memory`. The sequence *must* end with `finish_negotiation_round()`.
5.  **Framework:** Receives the sequence of tool calls from `agent_i` (e.g., as a list of structured objects).
6.  **Framework:** Parses and executes the tool calls sequentially:
    *   `send_message(...)`: Routes the message to the target agent's inbox staging area for the *next* round, adhering to the timing rules.
    *   `update_memory(...)`: Updates `agent_i`'s persistent `private_memory` store managed by the framework.
    *   `finish_negotiation_round()`: Marks `agent_i`'s turn complete for this round.
    *   Handles errors (invalid tool, bad parameters).
7.  **Framework:** Proceeds to the next agent in the sequence for the current round.

**End of Negotiation Phase:**

*   After the fixed number of rounds is completed, the Framework moves to the next phase.
*   The Framework ensures all messages sent in the *final* negotiation round are delivered (i.e., moved from staging to the main inbox accessible in the context bundle) before the Order Submission phase begins.

### 3.3. Order Submission Phase (Sequential)

This phase is for final decision-making and commitment.

1.  **Framework:** Iterates through each active agent sequentially.
2.  **Framework:** Assembles the Context Bundle for `agent_i`, including *all* messages from the completed Negotiation Phase.
3.  **Framework:** Calls `agent_i.take_turn(context_bundle, phase='ORDER_SUBMISSION')`.
4.  **Agent Logic:** Inside `take_turn`, the agent:
    *   Parses the `context_bundle`, including all negotiation messages.
    *   Executes its final internal reasoning.
    *   Outputs a sequence of **Action Tool Calls**. Key tools for this phase are `submit_order` and potentially `update_memory`. The sequence *must* end with `finish_orders()`.
5.  **Framework:** Receives the sequence of tool calls from `agent_i`.
6.  **Framework:** Parses and executes the tool calls:
    *   `submit_order(...)`: Stores the order string for `agent_i`.
    *   `update_memory(...)`: Updates `agent_i`'s persistent `private_memory`.
    *   `finish_orders()`: Marks `agent_i`'s order submission complete.
    *   Validates order count/format (or relies on the engine to do so later).
7.  **Framework:** Proceeds to the next agent in the sequence.

### 3.4. Adjudication

1.  **Framework:** Once orders are collected from all agents, compiles them into a single structure.
2.  **Framework:** Submits the complete set of orders to the Diplomacy Engine.
3.  **Framework:** Instructs the Diplomacy Engine to process the turn.
4.  **Framework:** Loops back to **Section 3.1** for the *next* game phase.

## 4. Agent Context Bundle (Input)

This is the information package provided by the Framework to the agent via `agent.take_turn(context_bundle, phase)`.

*   **`power_name` (str):** The agent's assigned power.
*   **`agent_instructions` (str):** Persistent high-level goals, personality traits, strategy guidance.
*   **`current_phase_info` (dict):**
    *   `phase_name` (str): e.g., "S1901M", "F1901R".
    *   `interaction_type` (str): e.g., "NEGOTIATION_ROUND_1", "ORDER_SUBMISSION".
    *   `negotiation_rounds_total` (int): Configured number of negotiation rounds.
*   **`public_game_state` (dict):** Snapshot of the current public state.
    *   `units` (dict): `{power: [(unit_type, location), ...], ...}`
    *   `centers` (dict): `{power: [center_location, ...], ...}`
    *   `influence` (dict): *(Optional)* Map of locations to influencing powers.
    *   `builds_disbands` (dict): `{power: count}` (Relevant in W phase).
    *   *Maybe other relevant state slices.*
*   **`history_summary` (dict):** Concisely summarized historical info (to manage context size).
    *   `last_phase_state` (dict): Key state elements (units, centers) from the *previous* phase.
    *   `recent_events` (list[str]): e.g., ["F1901M: Austria captured Serbia", "F1901M: France dislodged Germany from Burgundy"].
    *   `last_turn_messages_summary` (dict): `{other_power: summary_string, ...}` e.g., `{'England': 'Agreed support into Belgium', 'Germany': 'Rejected DMZ proposal'}`.
*   **`communication_inbox` (list[dict]):** Messages received by this agent *before* the current interaction step. Each message dict follows the format in Section 7.
*   **`private_memory_snippet` (dict):** *(Optional/Advanced)* Key persistent state retrieved by the framework from the agent's memory store (e.g., `{ 'trust_levels': {...}, 'active_plans': [...] }`).

**Context Management:** The Framework is responsible for generating the summaries (`history_summary`) to keep the context bundle manageable, especially for LLM-based agents.

## 5. Agent Action Tools (Output)

Instead of returning complex data structures, the agent's `take_turn` method outputs a sequence of desired actions by specifying tool calls. The Framework receives this sequence (e.g., as a list of JSON objects) and executes them.

**Core Tools:**

*   **`send_message(recipient: str, message_type: str, content: str)`:**
    *   *Purpose:* Send a message during a Negotiation Round.
    *   *Recipient:* `power_name` or `BROADCAST`.
    *   *Framework Action:* Adds message to recipient(s)' inbox staging area for the next round.
*   **`submit_order(order_string: str)`:**
    *   *Purpose:* Submit a single, binding order during the Order Submission Phase.
    *   *Framework Action:* Stores the order string for the agent. Validation may occur here or during engine submission.
*   **`update_memory(key: str, value: any)`:**
    *   *Purpose:* Allow the agent to persistently store or update its private state (beliefs, plans, trust levels, etc.).
    *   *Framework Action:* Updates the key-value pair in the agent's dedicated persistent memory store managed by the framework.
*   **`finish_negotiation_round()`:**
    *   *Purpose:* Signal that the agent has completed its actions for the current negotiation round turn.
    *   *Framework Action:* Marks agent as done for this step, allows framework to proceed to the next agent.
*   **`finish_orders()`:**
    *   *Purpose:* Signal that the agent has submitted all its orders for the current Order Submission Phase.
    *   *Framework Action:* Marks agent as done for the phase, allows framework to proceed.

**Example Agent Output (Order Submission Turn):**
```json
[
  { "tool_name": "submit_order", "arguments": { "order_string": "A PAR H" } },
  { "tool_name": "submit_order", "arguments": { "order_string": "A MAR S F SPA/SC - POR" } },
  { "tool_name": "submit_order", "arguments": { "order_string": "F BRE - MAO" } },
  { "tool_name": "update_memory", "arguments": { "key": "short_term_goal", "value": "Secure MAO" } },
  { "tool_name": "finish_orders", "arguments": {} }
]
```

## 6. Agent Responsibilities (Revisited)

*   Implement the `take_turn(context_bundle, phase)` method.
*   Parse the provided `context_bundle` to understand the current situation.
*   Perform internal reasoning based on context, `private_memory`, and agent's strategy.
*   Output a valid sequence of **Action Tool Calls** appropriate for the current `phase` (Negotiation or Order Submission).
*   Manage the *logic* of its `private_memory`, relying on the `update_memory` tool to persist changes.

## 7. Framework Responsibilities (Revisited)

*   Manage the main game loop and interact with the Diplomacy Engine.
*   Maintain the current interaction phase/round.
*   For each agent turn: **Assemble the appropriate Agent Context Bundle**, including necessary state slices and summaries.
*   Call the agent's `take_turn` method.
*   **Receive and parse the sequence of Action Tool Calls** from the agent.
*   **Execute the tool calls sequentially**, updating game state proxies (messages, orders) or the agent's persistent memory store.
*   Handle message routing and timing rules.
*   Validate tool calls and handle errors.
*   Collect final orders and submit them to the engine.
*   Manage the persistent `private_memory` store for each agent.

## 8. Communication Protocol

*(Structure unchanged, but emphasizes it's the format for messages handled by `send_message` tool and placed in the `communication_inbox` part of the context bundle)*
*   **Structure:**
    *   `sender`: `power_name`
    *   `recipient`: `power_name` or `BROADCAST`
    *   `message_type`: (e.g., `PROPOSAL`, `INFO`, `QUERY`, etc.)
    *   `content`: Payload (string, structured data).
    *   `turn`: Game turn/year/phase identifier.
    *   *Optional:* `round`: Negotiation round number within the phase.
*   **Transport:** Orchestrated by the Framework via `send_message` tool execution and context bundle assembly.

## 9. Key Considerations

*(No change from previous version, but the context/tool model highlights the importance of efficient context summarization and robust tool execution)*
*   **Sequential Bottleneck:** Total turn time depends on agent processing + framework overhead (context assembly, tool execution).
*   **No True Parallelism:** Agents react based on prior information.
*   **Framework Robustness:** Critical for managing context, tool execution, timing, and agent memory.
*   **Agent Complexity:** Agents need effective reasoning to utilize context and tools.
*   **Configuration:** Negotiation rounds, context summarization techniques.
*   **Tool Design:** Ensure tools are sufficient but minimal; avoid overly complex tools.
*   **Context Size:** Continuous monitoring and optimization needed, especially for LLMs.

This detailed model provides a concrete plan for implementing the agent interactions under the given constraints.
