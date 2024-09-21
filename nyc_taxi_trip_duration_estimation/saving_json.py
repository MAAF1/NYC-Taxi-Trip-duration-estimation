import os 
import json
import numpy as np
def save_json_with_numpy(data, filename):
    def numpy_to_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: numpy_to_json(v) for k, v in obj.items()}
        else:
            return obj

    try:
        # Check if the file exists
        if os.path.exists(filename):
            # Read existing data
            with open(filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            # Append new data to existing data
            existing_data[f'trial_{len(existing_data)}'] = data
            
            # Write updated data back to the file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, default=numpy_to_json, ensure_ascii=False, indent=4)
            
            print(f"New data appended to {filename}")
        else:
            # If file doesn't exist, create it with the new data
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({'trial_0': data}, f, default=numpy_to_json, ensure_ascii=False, indent=4)
            
            print(f"File {filename} created with new data")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")
    except json.JSONDecodeError:
        print(f"Invalid JSON format in file '{filename}'. Starting fresh.")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({'trial_0': data}, f, default=numpy_to_json, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 