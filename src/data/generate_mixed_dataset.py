import pandas as pd
import sys
import os
import multiprocessing
from tqdm import tqdm
import numpy as np
import copy

# Add src/data to path to import yosys_filter
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Try to import get_yosys_stats
try:
    from yosys_filter import get_yosys_stats
except ImportError:
    # Fallback if running from a different directory
    sys.path.append("./src/data")
    from yosys_filter import get_yosys_stats

def extract_verilog(row):
    # Try to find ground truth code
    # Check reward_model first (as seen in previous inspection)
    rm = row.get('reward_model')
    if isinstance(rm, dict) and 'ground_truth' in rm:
        return rm['ground_truth']
    if isinstance(rm, dict) and 'solution' in rm:
        return rm['solution']
    
    # Check top level
    if 'ground_truth' in row:
        return row['ground_truth']
    if 'solution' in row:
        return row['solution']
    
    # Check extra_info
    ei = row.get('extra_info')
    if isinstance(ei, dict) and 'ground_truth' in ei:
        return ei['ground_truth']
    
    return ""

def process_prompt(prompt):
    """Safely process prompt which might be a numpy array or list."""
    if isinstance(prompt, np.ndarray):
        # If it's an array of strings/objects, take the first one or join?
        # Assuming it's a single string wrapped in an array
        if prompt.size > 0:
            prompt = prompt.item(0) if prompt.size == 1 else str(prompt)
        else:
            prompt = ""
    elif isinstance(prompt, list):
         prompt = prompt[0] if prompt else ""
            
    if not isinstance(prompt, str):
        prompt = str(prompt)
        
    return prompt

def main():
    input_path = "./data/training_data/integral/codev_r1_rl_val.parquet"
    output_path = "./data/training_data/integral/codev_r1_rl_val_mixed.parquet"
    
    print(f"Loading {input_path}...")
    df = pd.read_parquet(input_path)
    total_len = len(df)
    print(f"Loaded {total_len} rows.")
    
    # Extract Verilog codes for difficulty analysis
    tasks = []
    for idx, row in df.iterrows():
        code = extract_verilog(row)
        tasks.append((idx, code))
        
    print("Running Yosys analysis to determine difficulty...")
    results = []
    num_workers = max(1, os.cpu_count())
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        for res in tqdm(pool.imap(get_yosys_stats, tasks, chunksize=50), total=total_len):
            results.append(res)
            
    # Calculate Difficulty Score
    # Score = (cells, ltp). Higher is more difficult.
    # None (Error) is treated as -1.
    scored_data = []
    for idx, cells, ltp, err in results:
        if cells is not None:
            score = (cells, ltp if ltp is not None else 0)
        else:
            score = (-1, -1)
        scored_data.append({'id': idx, 'score': score})
        
    # Sort by score descending (Most difficult first)
    scored_data.sort(key=lambda x: x['score'], reverse=True)
    
    # Select Top 50%
    top_n = int(total_len * 0.5)
    top_50_indices = set(item['id'] for item in scored_data[:top_n])
    
    print(f"Selected {len(top_50_indices)} difficult examples.")
    
    new_rows = []
    
    # Process Original Data
    print("Processing data...")
    for idx, row in tqdm(df.iterrows(), total=total_len):
        # 1. Original Instance (Always kept)
        original_item = row.to_dict()
        
        # Ensure extra_info is a fresh dict
        if isinstance(original_item.get('extra_info'), dict):
            original_item['extra_info'] = copy.deepcopy(original_item['extra_info'])
        else:
            original_item['extra_info'] = {}
            
        original_item['extra_info']['if_think'] = 'true'
        
        # Ensure prompt is a string in original item too, just in case
        original_item['prompt'] = process_prompt(original_item.get('prompt', ''))
        
        new_rows.append(original_item)
        
        # 2. Modified Instance (Only if in top 50% difficult)
        if idx in top_50_indices:
            modified_item = row.to_dict()
            if isinstance(modified_item.get('extra_info'), dict):
                modified_item['extra_info'] = copy.deepcopy(modified_item['extra_info'])
            else:
                modified_item['extra_info'] = {}
            
            # Update prompt: replace /think with /no_think
            raw_prompt = modified_item.get('prompt', '')
            prompt = process_prompt(raw_prompt)
            
            if prompt.endswith('/think'):
                modified_item['prompt'] = prompt[:-6] + '/no_think' 
            else:
                modified_item['prompt'] = prompt.replace('/think', '/no_think')
            
            # Set if_think = false
            modified_item['extra_info']['if_think'] = 'false'
            
            new_rows.append(modified_item)
            
    # Create new DataFrame
    new_df = pd.DataFrame(new_rows)
    print(f"New dataset size: {len(new_df)} (Original: {total_len})")
    
    # Save
    print(f"Saving to {output_path}...")
    new_df.to_parquet(output_path)
    print("Done.")

if __name__ == "__main__":
    main()
