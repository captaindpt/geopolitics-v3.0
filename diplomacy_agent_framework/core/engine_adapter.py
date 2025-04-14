from typing import List, Dict, Tuple, Optional

# Assuming the diplomacy engine is in the PYTHONPATH or installed
# We need to be careful about relative paths if running from different locations
try:
    from diplomacy import Game
    from diplomacy.engine.map import Map
    from diplomacy.utils.export import to_saved_game_format # For saving games
except ImportError as e:
    print(f"Error importing diplomacy library: {e}")
    print("Please ensure the 'diplomacy' engine library is installed or accessible in PYTHONPATH.")
    # Re-raise or exit if necessary, depending on desired robustness
    raise

from .data_structures import PublicGameState # Use our defined structure

class EngineAdapter:
    """Wraps the Diplomacy game engine to provide a stable interface for the framework."""

    def __init__(self, map_name: str = 'standard'):
        """Initializes the game engine."""
        print(f"Initializing Diplomacy game engine with map: {map_name}...")
        self.game = Game(map_name=map_name)
        self.powers = list(self.game.powers.keys())
        print(f"Game initialized. Phase: {self.get_current_phase()}")
        print(f"Powers: {self.powers}")

    def get_current_phase(self) -> str:
        """Returns the current game phase (e.g., 'S1901M')."""
        return self.game.get_current_phase()

    def is_game_done(self) -> bool:
        """Checks if the game has ended."""
        return self.game.is_game_done

    def get_all_powers(self) -> List[str]:
        """Returns the list of power names involved in the game."""
        return self.powers

    def get_public_game_state(self) -> PublicGameState:
        """Extracts the current public game state into our defined format using game.get_state()."""
        game_state_dict = self.game.get_state()
        # Example structure (based on typical game state representations, needs verification):
        # { 'name': 'S1901M', 
        #   'units': {'AUSTRIA': ['A BUD', 'A VIE', 'F TRI'], ... }, 
        #   'centers': {'AUSTRIA': ['BUD', 'VIE', 'TRI'], ... } 
        #   'builds': {'ENGLAND': {'count': 1, 'homes': [...]}}, ... }

        units_dict: Dict[str, List[Tuple[str, str]]] = {}
        centers_dict: Dict[str, List[str]] = {}
        builds_disbands_dict: Dict[str, int] = {}

        raw_units = game_state_dict.get('units', {})
        for power_name, unit_list in raw_units.items():
            units_dict[power_name] = []
            for unit_string in unit_list: # unit_string is like 'A BUD' or 'F TRI'
                parts = unit_string.split()
                if len(parts) == 2:
                     unit_type_char = parts[0] # 'A' or 'F'
                     location = parts[1]
                     # Handle potential coast specifiers like STP/SC - keep them with location for now
                     units_dict[power_name].append((unit_type_char, location))
                else:
                     print(f"Warning: Unexpected unit format '{unit_string}' in game state. Skipping.")

        raw_centers = game_state_dict.get('centers', {})
        for power_name, center_list in raw_centers.items():
            centers_dict[power_name] = center_list

        # Handle builds/disbands based on state dict structure
        current_phase = game_state_dict.get('name', '')
        if current_phase.endswith('A'): # Adjustment phase
             builds_info = game_state_dict.get('builds', {})
             for power_name, build_data in builds_info.items():
                  # 'count' usually represents number of builds allowed
                  builds_disbands_dict[power_name] = build_data.get('count', 0)
             # Note: Disbands might be represented differently or implicitly by negative build count
             # This needs verification based on how the diplomacy library handles disbands in get_state()
             # For now, we only capture positive build counts.

        return PublicGameState(
            phase_name=game_state_dict.get('name', 'UNKNOWN_PHASE'), # Extract phase name
            units=units_dict,
            centers=centers_dict,
            builds_disbands=builds_disbands_dict
        )

    def set_orders(self, power_name: str, orders: List[str]) -> None:
        """Sets the orders for a specific power for the current phase."""
        # Consider adding validation here if needed, or let the engine handle it.
        print(f"Adapter: Setting orders for {power_name}: {orders}")
        self.game.set_orders(power_name, orders)

    def process_turn(self) -> Dict[str, List[str]]:
        """Adjudicates the orders and returns the results for each power."""
        phase = self.get_current_phase()
        print(f"Adapter: Processing turn for phase {phase}...")
        
        # Store orders before processing (needed for context in results)
        original_orders_by_power = self.game.get_orders()
        
        # Process the turn
        self.game.process()
        print(f"Adapter: Turn processed. New phase: {self.get_current_phase()}")

        # Get order results
        # get_order_status() returns {power: {unit: [statuses]}} when called without args
        # For non-movement phases, it might return different structures, need to adapt
        # Let's focus on movement phase for now.
        formatted_results: Dict[str, List[str]] = {p: [] for p in self.powers}
        
        # Check if it was a movement phase where units map to orders
        if phase.endswith('M'):
            # game.result gives {unit: [statuses]} directly for the last processed phase
            order_results = self.game.result
            for power_name, power_orders_list in original_orders_by_power.items():
                # Reconstruct the original full order string to match it with results
                power_original_orders_dict = {}
                for order_str in power_orders_list:
                    parts = order_str.split() # e.g., ['A', 'PAR', 'H'] or ['F', 'BRE', 'S', 'A', 'PAR', '-', 'PIC']
                    if len(parts) >= 2:
                        unit_key = f"{parts[0]} {parts[1]}" # e.g., "A PAR"
                        power_original_orders_dict[unit_key] = order_str

                power_state = self.game.get_state()
                power_units = [u.replace('*','') for u in power_state['units'].get(power_name, [])] # Units at START of phase 
                
                for unit in power_units:
                    status_list = order_results.get(unit, [])
                    original_order = power_original_orders_dict.get(unit, f"{unit} [Default Hold - Not Found]") # Find the submitted order string
                    if not status_list: # Empty list means success
                        formatted_results[power_name].append(f"Order '{original_order}': Success")
                    else:
                        # Join multiple statuses like 'cut', 'void'
                        status_str = ", ".join(status_list).upper()
                        formatted_results[power_name].append(f"Order '{original_order}': FAILED ({status_str})")
            
        # TODO: Handle results for Retreat (R) and Adjustment (A) phases if needed
        # The structure of game.result might differ, or we might need game.get_order_status()

        return formatted_results

    def get_final_centers(self) -> Optional[Dict[str, List[str]]]:
         """Gets the final center count if the game is done."""
         if self.is_game_done():
              # game.get_centers() returns {power: [centers]} when called without args
              return self.game.get_centers()
         return None

    def save_game(self, output_path: str) -> None:
        """Saves the current game state to a file."""
        print(f"Adapter: Saving game to {output_path}...")
        try:
            # to_saved_game_format requires the game object
            game_data = to_saved_game_format(self.game)
            with open(output_path, 'w') as f:
                import json
                json.dump(game_data, f, indent=4)
            print("Adapter: Game saved successfully.")
        except Exception as e:
            print(f"Adapter: Error saving game: {e}")

    # Add any other necessary methods to interact with the engine 