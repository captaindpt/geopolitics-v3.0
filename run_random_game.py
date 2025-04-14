import random
# We might need to adjust this import based on where we run the script from
# Assuming we run from the workspace root, and the engine is in 'diplomacy_engine'
# However, the engine's internal structure might expect to be run from its own root or be installed.
# Let's try the direct import first, assuming it might work if run correctly or if added to path.
from diplomacy import Game
from diplomacy.utils.export import to_saved_game_format
import json
import time # Added for the message example if we uncomment it

# Creating a game
# Alternatively, a map_name can be specified as an argument. e.g. Game(map_name='pure')
print("Creating game...")
game = Game()
print(f"Game created. Initial phase: {game.get_current_phase()}")

turn_count = 0
while not game.is_game_done:
    current_phase = game.get_current_phase()
    print(f"--- Processing Phase: {current_phase} ---")

    # Getting the list of possible orders for all locations
    possible_orders = game.get_all_possible_orders()

    orders_set = False
    # For each power, randomly sampling a valid order
    for power_name in game.powers:
        power_orders = []
        orderable_locations = game.get_orderable_locations(power_name)
        if orderable_locations:
            # Filter locations that actually have possible orders
            valid_locations = [loc for loc in orderable_locations if possible_orders.get(loc)]
            power_orders = [random.choice(possible_orders[loc]) for loc in valid_locations if possible_orders[loc]] # Ensure list is not empty

        if power_orders:
            print(f"Setting orders for {power_name}: {power_orders}")
            game.set_orders(power_name, power_orders)
            orders_set = True
        else:
            print(f"No orderable locations/orders for {power_name} this phase.")
            game.set_orders(power_name, []) # Explicitly set empty orders if none possible


    # Only process if any orders were possible/set, otherwise could loop infinitely
    if not orders_set and not game.is_game_done:
         print("No orders possible for any power. Checking if game should end.")
         # If no orders are possible and game isn't done, something might be wrong,
         # but process() should handle phase transition (e.g. retreats needed)
         # Let's add a safeguard just in case
         if turn_count > 500: # Arbitrary limit to prevent infinite loops
             print("Warning: Exceeded turn limit without game completion. Breaking.")
             break

    # Messages can be sent locally with game.add_message
    # e.g. game.add_message(Message(sender='FRANCE',
    #                               recipient='ENGLAND',
    #                               message='This is a message',
    #                               phase=game.get_current_phase(), # Corrected self reference
    #                               time_sent=int(time.time())))

    # Processing the game to move to the next phase
    print("Processing game phase...")
    game.process()
    print(f"Phase processed. New phase: {game.get_current_phase()}")
    turn_count += 1 # Increment turn count

print(f"--- Game Over ---")
# print(f"Winner(s): {game.get_winners()}")
print(f"Game done: {game.is_game_done}")
print(f"Final Phase: {game.get_current_phase()}")
try:
    final_centers = game.get_centers()
    print(f"Final Center Counts: {final_centers}")
    # Determine winner based on center count (optional, simple majority check)
    if final_centers:
        winners = [power for power, centers in final_centers.items() if len(centers) >= 18]
        if winners:
            print(f"Power(s) with >= 18 centers: {winners}")
        else:
            print("No single power achieved solo victory (>= 18 centers).")
except AttributeError:
    print("Could not retrieve final center counts (get_centers() might not exist or failed).")


# Exporting the game to disk to visualize (game is appended to file)
# The original example uses append mode implicitly via the function. Let's make it explicit.
output_filename = 'game.json'
print(f"Exporting game to {output_filename}...")
try:
    # The function expects a file object or path. Path is simpler.
    # Overwrite the file each time for this test script.
    with open(output_filename, 'w') as f:
         json.dump(to_saved_game_format(game), f, indent=4)
    print("Game exported successfully.")
except Exception as e:
    print(f"Error exporting game: {e}") 