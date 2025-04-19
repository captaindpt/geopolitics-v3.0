# Phase 3 Observations (Run 2 - Single Execution)

Based on the terminal output from the corrected single run:

**Overall Sense:**
*   Agent "Thoughts" often reflect standard opening strategic considerations (e.g., Austria vs. Turkey, England naval control, France Iberia).
*   However, the translation of these thoughts into actual **orders** is frequently poor and nonsensical.

**Effectiveness / Winning Strategy:**
*   It's difficult to say the agents are coherently *trying* to win at this stage.
*   Execution of stated intentions (expansion, alliances) is deeply flawed.

**Specific Issues & Examples:**
*   **Nonsensical/Impossible Orders:** Many orders violate geography or basic unit rules.
    *   Germany: `F KIE - BUD` (Fleet to landlocked Budapest), `A MUN S A BER - BALT` (Army supporting into sea).
    *   Russia: `A WAR LVP` (Army Warsaw->Liverpool?), `F SEV BALT` (Fleet Sevastopol->Baltic?), `F STP/SC NWP` (Fleet St. Petersburg->Northwest Passage?).
    *   Turkey: `F ANK - SZE` (Fleet Ankara->Suez?), `A SMY - EGY` (Army Smyrna->Egypt?).
    *   England: Bounced own fleets in English Channel (`F EDI - ENG`, `F LON - ENG` in F1901M).
*   **Passive/Weak Moves:** Some powers made unproductive moves.
    *   Austria: Self-blocking opening (`A BUD - VIE`, `F TRI - BUD`) repeated for 3 phases.
    *   England: Unusual `F EDI LVP` opening.
    *   Italy: Aggressive but unconventional `F NAP - ADR` S `A VEN` repeated for 3 phases.
*   **Inconsistent Logic:** Russia fluctuated between impossible orders and standard moves.
*   **France Anomaly:** Showed *some* coherence (standard ENG move, later BUR & NTH towards ENG) but repeatedly tried messaging non-existent powers (Spain, Portugal).
*   **Lack of Adaptation (in 3 phases):** Little change in strategy; Austria and Italy repeated the exact same peculiar orders.

**Conclusion:**
*   Agents generate somewhat relevant high-level strategic text ("Thoughts").
*   Significant failure in translating thoughts into **valid, sensible, and strategically effective actions (orders)**.
*   A clear disconnect exists between planning/thought generation and order execution/validation.

## Communication Flow & Improvements

**Current Limitation:**
*   The current framework uses two negotiation rounds per phase.
*   Agents send messages simultaneously in each round and only receive messages from the *previous* round at the *start* of the current round.
*   **Crucially, agents cannot react to messages sent by others *within the same round*.** This creates a delay and prevents real-time back-and-forth dialogue.
*   The second negotiation round currently seems underutilized, with agents not showing significant adaptation based on messages received from Round 1.

**Potential Improvements Discussed:**
1.  **Enhance Round 2 Prompting:** Modify the Round 2 prompt to explicitly instruct agents to react to messages received from Round 1, potentially making the second round more meaningful.
2.  **Increase Rounds:** Add more negotiation rounds (3+) to allow more delayed exchanges. (Potential downside: slows game significantly).
3.  **Simplify to One Round:** Reduce complexity by having only one negotiation round per movement phase. Focuses on fixing core action generation first.
4.  **(Advanced) Iterative Negotiation:** Design a more complex intra-phase loop for real-time message exchange. (High implementation complexity).

**Next Steps Consideration:** Options 1 or 3 seem most practical initially.

## Improving Agent Reasoning ("Thinking")

**Problem:** Agents generate plausible high-level thoughts but fail to translate them into valid/sensible orders. A reasoning step is missing or ineffective.

**Approaches Discussed:**
1.  **Injecting a "Thinking Token":**
    *   Concept: Use a special token to trigger an internal reasoning mode in the LLM, potentially requiring model fine-tuning or specific features.
    *   Pros: Explicit control, potentially cleaner parsing of thoughts.
    *   Cons: Requires model support, framework complexity, potentially brittle.
2.  **Prompt Engineering (Chain-of-Thought / Guided Reasoning):**
    *   Concept: Craft the prompt to explicitly guide the LLM through steps: analyze state, list valid moves, evaluate strategy, format orders.
    *   Pros: Works with most capable LLMs, easier implementation (modify prompt text), flexible.
    *   Cons: Relies on LLM's instruction following, potentially longer prompts/latency.

**Recommendation:**
*   Given the use of a general TGI endpoint, **Prompt Engineering (Approach 2)** is the most practical and accessible starting point.
*   Focus on modifying the prompts, especially for the `ORDER_SUBMISSION` phase, to enforce a step-by-step reasoning process before outputting orders.
