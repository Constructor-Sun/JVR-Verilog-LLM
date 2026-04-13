import pandas as pd
import os
import ast
import json
import glob

def fix_prompt_format(prompt_item):
    """
    Convert prompt item to List[Dict].
    Handles:
    - str (JSON or Python dict string) -> parse -> [dict]
    - dict -> [dict]
    - list -> list (assume already correct)
    """
    if isinstance(prompt_item, list):
        return prompt_item
    
    if isinstance(prompt_item, dict):
        return [prompt_item]
    
    if isinstance(prompt_item, str):
        # Try parsing as JSON first
        try:
            d = json.loads(prompt_item)
            if isinstance(d, dict):
                return [d]
            if isinstance(d, list):
                return d
        except json.JSONDecodeError:
            pass
            
        # Try parsing as Python literal
        try:
            d = ast.literal_eval(prompt_item)
            if isinstance(d, dict):
                return [d]
            if isinstance(d, list):
                return d
        except (ValueError, SyntaxError):
            print(f"Error parsing prompt: {prompt_item[:100]}...")
            return prompt_item # Return original if failed

    return prompt_item

def fix_file(file_path):
    print(f"Processing {file_path}...")
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return

    if "prompt" not in df.columns:
        print(f"No 'prompt' column in {file_path}. Skipping.")
        return

    # Check type of first element
    first_prompt = df["prompt"].iloc[0]
    print(f"Original type: {type(first_prompt)}")
    print(f"Original sample: {str(first_prompt)[:100]}...")

    # Apply fix
    df["prompt"] = df["prompt"].apply(fix_prompt_format)

    # Verify
    new_first_prompt = df["prompt"].iloc[0]
    print(f"New type: {type(new_first_prompt)}")
    print(f"New sample: {new_first_prompt}")
    
    if isinstance(new_first_prompt, list) and len(new_first_prompt) > 0 and isinstance(new_first_prompt[0], dict):
        print("Verification successful: prompt is List[Dict]")
    else:
        print("Verification warning: prompt might not be correct format")

    # Save
    df.to_parquet(file_path)
    print(f"Saved {file_path}")

def main():
    # Base directory for data
    base_dir = "./final_project/data/training_data/integral"
    
    # Find all parquet files that might need fixing
    # User mentioned codev_r1_rl_val_mixed.parquet
    # We should probably look for others too.
    files = [os.path.join(base_dir, f) for f in ["codev_r1_rl_train_mixed.parquet", "codev_r1_rl_val_mixed.parquet"]]
    
    for f in files:
        if "mixed" in f: # Prioritize the mixed ones as per user context, but maybe all of them?
            fix_file(f)
        else:
            # Check non-mixed ones too?
            # Let's inspect them lightly or just fix them. 
            # If the format is already correct, the function handles it.
            fix_file(f)

if __name__ == "__main__":
    main()
