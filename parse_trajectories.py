import xml.etree.ElementTree as ET
from collections import defaultdict
import sys
import numpy as np

# --- NEW: Grid Discretization Functions ---

def create_grid_from_trajectories(trajectories, cell_size=50):
    """
    Analyzes all trajectories to find the map boundaries and creates a grid.

    Args:
        trajectories (dict): The dictionary of vehicle trajectories.
        cell_size (int): The width and height of each grid cell in meters.

    Returns:
        A tuple containing:
        - min_x, min_y (float): The bottom-left corner of the grid.
        - cols, rows (int): The number of columns and rows in the grid.
    """
    # Flatten all points from all trajectories to find the boundaries
    all_points = [point for traj in trajectories.values() for point in traj]
    
    if not all_points:
        return 0, 0, 0, 0
        
    min_x = min(p[0] for p in all_points)
    max_x = max(p[0] for p in all_points)
    min_y = min(p[1] for p in all_points)
    max_y = max(p[1] for p in all_points)
    
    # Calculate the number of columns and rows needed
    cols = int(np.ceil((max_x - min_x) / cell_size))
    rows = int(np.ceil((max_y - min_y) / cell_size))
    
    print(f"\nGrid created with {cols} columns and {rows} rows (cell size: {cell_size}m).")
    
    return min_x, min_y, cols, rows

def get_cell_id(x, y, min_x, min_y, cols, cell_size):
    """
    Maps an (x, y) coordinate to a unique integer cell ID.

    Args:
        x, y (float): The coordinate to map.
        min_x, min_y (float): The bottom-left corner of the grid.
        cols (int): The total number of columns in the grid.
        cell_size (int): The size of each grid cell.

    Returns:
        An integer representing the unique cell ID.
    """
    col = int((x - min_x) / cell_size)
    row = int((y - min_y) / cell_size)
    
    # Calculate a unique ID for the (row, col) pair
    cell_id = row * cols + col
    return cell_id

# --- (The parsing function from before is unchanged) ---

def parse_fcd_output(xml_file):
    trajectories = defaultdict(list)
    try:
        for _, elem in ET.iterparse(xml_file):
            if elem.tag == 'vehicle':
                vehicle_id = elem.get('id')
                x, y = elem.get('x'), elem.get('y')
                if vehicle_id and x and y:
                    trajectories[vehicle_id].append((float(x), float(y)))
            elem.clear()
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except FileNotFoundError:
        print(f"Error: '{xml_file}' not found.")
        sys.exit()
    return trajectories

# --- Main execution ---
if __name__ == "__main__":
    FCD_FILE = 'fcd-output.xml'
    
    print(f"Parsing trajectories from '{FCD_FILE}'...")
    # 1. Get trajectories as coordinates
    vehicle_trajectories_coords = parse_fcd_output(FCD_FILE)
    
    if vehicle_trajectories_coords:
        print(f"Successfully parsed {len(vehicle_trajectories_coords)} vehicle trajectories.")
        
        # 2. Create the grid based on these coordinates
        min_x, min_y, cols, rows = create_grid_from_trajectories(vehicle_trajectories_coords, cell_size=50)

        # 3. Convert coordinate trajectories to cell ID trajectories
        discretized_trajectories = defaultdict(list)
        for vehicle_id, traj in vehicle_trajectories_coords.items():
            cell_sequence = []
            for x, y in traj:
                cell_id = get_cell_id(x, y, min_x, min_y, cols, 50)
                # To avoid long runs of the same cell if a car is stopped
                if not cell_sequence or cell_sequence[-1] != cell_id:
                    cell_sequence.append(cell_id)
            discretized_trajectories[vehicle_id] = cell_sequence

        print("\n--- Discretization Complete ---")
        
        # 4. Inspect the result for the first vehicle
        if discretized_trajectories:
            first_vehicle_id = list(discretized_trajectories.keys())[0]
            first_discretized_traj = discretized_trajectories[first_vehicle_id]
            
            print(f"\nExample: Trajectory for Vehicle '{first_vehicle_id}' as Cell IDs")
            print(f"Original number of points: {len(vehicle_trajectories_coords[first_vehicle_id])}")
            print(f"Discretized sequence length: {len(first_discretized_traj)}")
            print(f"First 10 Cell IDs in sequence: {first_discretized_traj[:10]}")