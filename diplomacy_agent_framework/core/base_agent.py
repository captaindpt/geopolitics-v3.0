import abc
from typing import List

# Import the necessary data structures
from .data_structures import AgentContextBundle, ActionToolCall

class BaseAgent(abc.ABC):
    """Abstract base class for all Diplomacy agents."""

    def __init__(self, power_name: str):
        """Initialize the agent with its assigned power."""
        self.power_name = power_name
        # Note: Agent's persistent private memory is managed by the Framework,
        # but the agent's logic operates on it conceptually.

    @abc.abstractmethod
    def take_turn(self, context_bundle: AgentContextBundle, run_log_dir: str = None) -> List[ActionToolCall]:
        """
        Processes the given context and determines the agent's actions for its current turn.

        This method encapsulates the agent's thinking process for both negotiation
        rounds and the final order submission phase, differentiated by the
        `interaction_type` within the `context_bundle.current_phase_info`.

        Args:
            context_bundle: An object containing all relevant information (game state,
                            messages, history summaries, instructions, etc.) needed
                            for the agent to make a decision.
            run_log_dir: Optional path to a directory for run-specific logging.

        Returns:
            A list of ActionToolCall objects representing the sequence of actions
            the agent wishes to perform (e.g., send messages, submit orders,
            update memory). The sequence MUST end with either
            `finish_negotiation_round()` or `finish_orders()` depending on the phase.
        """
        pass

    # Potential future helper methods could be added here, e.g.,
    # - load_memory(snapshot)
    # - get_memory_snapshot() 