from typing import List, Tuple

from diplomacy_agent_framework.core.base_agent import BaseAgent
from diplomacy_agent_framework.core.data_structures import AgentContextBundle, ActionToolCall

class HoldAgent(BaseAgent):
    """A very simple agent that holds all its units in place."""

    def take_turn(self, context_bundle: AgentContextBundle) -> List[ActionToolCall]:
        """
        Generates actions for the agent's turn.

        - If in a negotiation round, does nothing except finish the round.
        - If in the order submission phase, generates HOLD orders for all units.
        """
        actions: List[ActionToolCall] = []
        interaction_type = context_bundle.current_phase_info.interaction_type

        if interaction_type == "ORDER_SUBMISSION":
            # Generate HOLD orders for all owned units
            my_units: List[Tuple[str, str]] = context_bundle.public_game_state.units.get(self.power_name, [])
            for unit_type, location in my_units:
                order_string = f"{unit_type} {location} H"
                actions.append(ActionToolCall(tool_name="submit_order", arguments={"order_string": order_string}))

            # Finish the order submission phase
            actions.append(ActionToolCall(tool_name="finish_orders"))

        elif interaction_type.startswith("NEGOTIATION_ROUND_"):
            # Do nothing during negotiation, just finish the turn
            actions.append(ActionToolCall(tool_name="finish_negotiation_round"))

        else:
            # Should not happen with current design, but handle defensively
            print(f"Warning: {self.power_name} received unknown interaction type: {interaction_type}")
            # Default to finishing the phase it thinks it might be in
            if interaction_type == "ORDER_SUBMISSION":
                 actions.append(ActionToolCall(tool_name="finish_orders"))
            else:
                 actions.append(ActionToolCall(tool_name="finish_negotiation_round"))


        return actions 