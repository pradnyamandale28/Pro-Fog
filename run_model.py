import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
import sys
import random

# --- (Functions from Step 3: Parsing and Discretization) ---

def parse_fcd_output(xml_file):

    trajectories = defaultdict(list)
    try:
        for _, elem in ET.iterparse(xml_file):
            if elem.tag == 'vehicle':
                vehicle_id, x, y = elem.get('id'), elem.get('x'), elem.get('y')
                if vehicle_id and x and y:
                    trajectories[vehicle_id].append((float(x), float(y)))
            elem.clear()
    except (ET.ParseError, FileNotFoundError):
        print(f"Error reading '{xml_file}'. Please ensure it exists and is valid.")
        sys.exit()
    return trajectories

def discretize_all_trajectories(trajectories_coords, cell_size=50):
    
    all_points = [point for traj in trajectories_coords.values() for point in traj]
    if not all_points: return {}, 0, 0, 0, 0
    min_x, max_x = min(p[0] for p in all_points), max(p[0] for p in all_points)
    min_y, max_y = min(p[1] for p in all_points), max(p[1] for p in all_points)
    cols = int(np.ceil((max_x - min_x) / cell_size))
    
    discretized_trajectories = defaultdict(list)
    for v_id, traj in trajectories_coords.items():
        cell_sequence = [int(((p[1] - min_y) / cell_size)) * cols + int(((p[0] - min_x) / cell_size)) for p in traj]
        # Remove consecutive duplicates
        if cell_sequence:
            discretized_trajectories[v_id] = [v for i, v in enumerate(cell_sequence) if i == 0 or v != cell_sequence[i-1]]
            
    return discretized_trajectories

# --- NEW: Markov Chain Model ---

class MarkovChain:
    def __init__(self):
        """Initializes the Markov Chain model."""
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.prediction_map = {}

    def fit(self, trajectories):
        """
        Trains the model by counting transitions between states.
        
        Args:
            trajectories (list of lists): A list of discretized trajectories.
        """
        print("\nTraining Markov Chain model...")
        for traj in trajectories:
            # Iterate through each pair of consecutive states in the trajectory
            for i in range(len(traj) - 1):
                current_state = traj[i]
                next_state = traj[i+1]
                self.transition_matrix[current_state][next_state] += 1
        
        # Create a simple prediction map: for each state, find the most likely next state
        for current_state, next_states in self.transition_matrix.items():
            most_likely_next_state = max(next_states, key=next_states.get)
            self.prediction_map[current_state] = most_likely_next_state
        print("Training complete.")

    def predict(self, current_state):
        """
        Predicts the next state given the current state.
        
        Returns:
            The predicted next state, or None if the current state was never seen.
        """
        return self.prediction_map.get(current_state)

# --- Main execution ---
if __name__ == "__main__":
    # 1. Load and Preprocess Data
    FCD_FILE = 'fcd-output.xml'
    print(f"--- Step 1: Loading and Preprocessing Data from '{FCD_FILE}' ---")
    trajectories_coords = parse_fcd_output(FCD_FILE)
    discretized_trajectories = discretize_all_trajectories(trajectories_coords)
    
    # Get a list of all valid trajectories (sequences of cell IDs)
    all_sequences = [seq for seq in discretized_trajectories.values() if len(seq) > 1]
    print(f"Found {len(all_sequences)} valid trajectories for modeling.")

    # 2. Split Data into Training and Testing Sets
    print("\n--- Step 2: Splitting Data ---")
    random.seed(42) # for reproducibility
    random.shuffle(all_sequences)
    
    split_index = int(len(all_sequences) * 0.80) # 80% for training, 20% for testing
    train_sequences = all_sequences[:split_index]
    test_sequences = all_sequences[split_index:]
    
    print(f"{len(train_sequences)} sequences for training.")
    print(f"{len(test_sequences)} sequences for testing.")
    
    # 3. Train the Model
    model = MarkovChain()
    model.fit(train_sequences)
    
    # 4. Evaluate the Model
    print("\n--- Step 3: Evaluating Model Performance ---")
    correct_predictions = 0
    total_predictions = 0
    
    for test_traj in test_sequences:
        for i in range(len(test_traj) - 1):
            current_cell = test_traj[i]
            actual_next_cell = test_traj[i+1]
            
            # Get the model's prediction
            predicted_next_cell = model.predict(current_cell)
            
            if predicted_next_cell is not None:
                if predicted_next_cell == actual_next_cell:
                    correct_predictions += 1
                total_predictions += 1
                
    # Calculate accuracy
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"\n--- Evaluation Results ---")
        print(f"Total Predictions Made on Test Set: {total_predictions}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Model Accuracy: {accuracy:.2f}%")
    else:
        print("No predictions were made. The test set might be too small or contain only new states.")
