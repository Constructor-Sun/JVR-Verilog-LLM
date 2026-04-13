import pandas as pd
import numpy as np
import sys

def process_prompt(prompt):
    """Safely process prompt which might be a numpy array or list."""
    if isinstance(prompt, np.ndarray):
        if prompt.size > 0:
            prompt = prompt.item(0) if prompt.size == 1 else str(prompt)
        else:
            prompt = ""
    elif isinstance(prompt, list):
         prompt = prompt[0] if prompt else ""
            
    if not isinstance(prompt, str):
        prompt = str(prompt)
        
    return prompt

def check_ratio(parquet_path):
    print(f"Loading {parquet_path}...")
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    total = len(df)
    print(f"Total rows: {total}")

    no_think_count = 0
    think_count = 0
    other_count = 0

    for idx, row in df.iterrows():
        prompt = process_prompt(row['prompt'])
        
        if prompt.endswith('/no_think'):
            no_think_count += 1
        elif prompt.endswith('/think'):
            think_count += 1
        else:
            other_count += 1
            if other_count <= 5:
                print(f"Sample 'other' prompt ending: ...{prompt[-20:]}")

    print(f"\nCounts:")
    print(f"  /no_think: {no_think_count}")
    print(f"  /think:    {think_count}")
    print(f"  Other:     {other_count}")

    if total > 0:
        ratio = no_think_count / total
        print(f"\nRatio of /no_think: {ratio:.4f}")
        
        expected_ratio = 1/3
        print(f"Expected Ratio (1/3): {expected_ratio:.4f}")
        
        if abs(ratio - expected_ratio) < 0.01:
            print("SUCCESS: Ratio is approximately 1/3.")
        else:
            print("WARNING: Ratio deviates from 1/3.")
    else:
        print("Total is 0.")

if __name__ == "__main__":
    path = "./data/training_data/integral/codev_r1_rl_val_mixed.parquet"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    check_ratio(path)
