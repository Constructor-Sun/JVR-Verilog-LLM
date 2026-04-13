import pandas as pd
import numpy as np
import sys

def inspect_prompt(parquet_path):
    print(f"Loading {parquet_path}...")
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Find a row where prompt is a numpy array
    found_example = False
    
    print("\nSearching for numpy array prompt...")
    for idx, row in df.iterrows():
        raw_prompt = row['prompt']
        
        if isinstance(raw_prompt, np.ndarray):
            print(f"\n[FOUND] Row {idx} has prompt of type: {type(raw_prompt)}")
            print(f"Numpy Array Shape: {raw_prompt.shape}")
            print(f"Numpy Array Dtype: {raw_prompt.dtype}")
            print("\n--- Full Numpy Array Representation ---")
            # Use repr to show the array structure explicitly
            print(repr(raw_prompt))
            
            print("\n--- Element Inspection ---")
            if raw_prompt.size > 0:
                element = raw_prompt.item(0)
                print(f"Element Type: {type(element)}")
                print(f"Element Content (first 500 chars):\n{str(element)[:500]}...")
            
            found_example = True
            break
    
    if not found_example:
        print("No numpy array prompts found in the first pass inspection.")
        # Check first row anyway
        first_prompt = df.iloc[0]['prompt']
        print(f"\nFirst Row Type: {type(first_prompt)}")
        print(repr(first_prompt))

if __name__ == "__main__":
    path = "./data/training_data/integral/codev_r1_rl_train.parquet"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    inspect_prompt(path)
