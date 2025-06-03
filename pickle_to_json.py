import os
import pickle
import json

def convert_pickle_to_json(checkpoint_dir, json_path):
    """Convert a data_split.pkl file into a JSON file"""

    data_split_path = os.path.join(checkpoint_dir, 'data_split.pkl')

    if not os.path.exists(data_split_path):
        raise FileNotFoundError(f"No pickle file found at {data_split_path}")

    with open(data_split_path, 'rb') as f:
        data_split = pickle.load(f)

    # Convert non-serializable objects to lists (e.g., numpy arrays)
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(i) for i in obj]
        elif hasattr(obj, 'tolist'):  # for numpy arrays
            return obj.tolist()
        else:
            return obj

    serializable_data_split = make_serializable(data_split)

    with open(json_path, 'w') as f:
        json.dump(serializable_data_split, f, indent=2)

# Example usage:
convert_pickle_to_json('./', 'data_split.json')
