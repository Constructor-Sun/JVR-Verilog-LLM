import json
import re
import subprocess
import tempfile
import os
import sys
import multiprocessing
from tqdm import tqdm

def get_yosys_stats(args):
    """
    Worker function to compute stats for a single Verilog code snippet.
    args: (index, verilog_code)
    Returns: (index, cells_count, ltp, error_msg)
    """
    index, verilog_code = args
    
    if not verilog_code:
        return index, None, None, "Empty code"

    # Create a temporary file for the Verilog code
    # Use a unique suffix to avoid collisions in parallel execution
    # We use /tmp or a specific temp dir to ensure we have write permissions and speed
    try:
        # Use directory='.' if we want to be safe about permissions in current env, 
        # or default temp dir. The error "Access denied" in Write tool suggested 
        # we might have restriction on where we write files? 
        # But tempfile.NamedTemporaryFile uses /tmp usually.
        # Let's try using local directory for temp files to be safe, or just standard temp.
        # Standard temp is usually fine for execution.
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{index}.v', delete=False) as tmp:
            tmp.write(verilog_code)
            tmp_path = tmp.name
    except Exception as e:
        return index, None, None, f"Temp file error: {e}"

    # Yosys command
    # -auto-top might fail if multiple modules are present and not clear top.
    # flatten is important for accurate cell count.
    yosys_cmd = f"read_verilog {tmp_path}; hierarchy -auto-top; proc; opt; flatten; opt; stat; ltp"
    
    try:
        # Run yosys
        result = subprocess.run(
            ['yosys', '-p', yosys_cmd],
            capture_output=True,
            text=True,
            timeout=30  # Timeout per file
        )
        
        output = result.stdout
        stderr = result.stderr
        
        # Clean up temp file immediately
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        if result.returncode != 0:
            return index, None, None, "Yosys error"

        # Parse cells count
        cells_count = None
        cells_match = re.search(r'^\s*(\d+)\s+cells', output, re.MULTILINE)
        if cells_match:
            cells_count = int(cells_match.group(1))
        
        if cells_count is None:
            cells_match_alt = re.search(r'Number of cells:\s+(\d+)', output)
            if cells_match_alt:
                cells_count = int(cells_match_alt.group(1))

        # Parse longest topological path
        ltp = None
        ltp_match = re.search(r'Longest topological path in .* \(length=(\d+)\)', output)
        if ltp_match:
            ltp = int(ltp_match.group(1))
            
        return index, cells_count, ltp, None

    except subprocess.TimeoutExpired:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return index, None, None, "Timeout"
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return index, None, None, str(e)

def process_line(line):
    try:
        data = json.loads(line)
        response = data.get('response', '')
        # Clean response to get code
        code = response.replace('```verilog', '').replace('```', '').strip()
        return code
    except:
        return ""

def main():
    nothink_file = './data/training_data/integral/codev_r1_sft_nothink_filtered.jsonl'
    think_file = './data/training_data/integral/codev_r1_sft_think_filtered.jsonl'
    output_file = './data/training_data/integral/codev_r1_sft_mixed_filtered.jsonl'
    
    if not os.path.exists(nothink_file) or not os.path.exists(think_file):
        print("Error: Input files not found.")
        return

    print("Reading nothink file to extract codes...")
    tasks = []
    with open(nothink_file, 'r') as f:
        for i, line in enumerate(f):
            code = process_line(line)
            tasks.append((i, code))
    
    total_tasks = len(tasks)
    print(f"Total examples to process: {total_tasks}")
    
    # Run Yosys in parallel
    print("Running Yosys analysis...")
    results = []
    # Use slightly fewer processes than CPU count to keep system responsive, or max if safe.
    # On a dedicated server, CPU count is fine.
    num_workers = max(1, os.cpu_count()) 
    print(f"Using {num_workers} workers.")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use imap_unordered for better progress bar updates
        for res in tqdm(pool.imap(get_yosys_stats, tasks, chunksize=50), total=total_tasks):
            results.append(res)
    
    # results is a list of (index, cells, ltp, error)
    # We sort results by index to ensure alignment, though not strictly necessary if we use a dict
    # results.sort(key=lambda x: x[0]) 
    
    # Calculate Complexity and Split
    # Strategy: 
    # Complexity = Cells. If Cells is None, Complexity = Infinity (Hard)
    # Tie-breaker: LTP
    
    valid_data = []
    for idx, cells, ltp, err in results:
        if cells is not None:
            # Prefer simpler (fewer cells). 
            # If cells are equal, prefer shorter path (lower LTP).
            score = (cells, ltp if ltp is not None else float('inf'))
            valid_data.append({'id': idx, 'score': score, 'cells': cells, 'ltp': ltp})
        else:
            # Error -> Treat as Hard (Infinity)
            # We assign a score that is larger than any valid score
            valid_data.append({'id': idx, 'score': (float('inf'), float('inf')), 'cells': None, 'ltp': None})

    import random

    # Sort by score
    # Python tuples compare element by element: (cells, ltp)
    valid_data.sort(key=lambda x: x['score'])
    
    alpha = 0.2
    cutoff_index = int(total_tasks * alpha)
    
    # Initial split based on complexity
    easy_pool = valid_data[:cutoff_index]
    hard_pool = valid_data[cutoff_index:]
    
    # Mixing Logic
    # The user wants "nothink" samples to come from both Easy and Hard parts,
    # but the total count of "nothink" must remain alpha * Total (which is size of easy_pool).
    # We achieve this by swapping a portion of the assignments.
    # We will take X% of the "nothink" budget and assign it to "hard" examples,
    # which means we must take X% of the "easy" examples and assign them to "think".
    
    mix_ratio = 0.2 # 20% of nothink data comes from hard pool
    n_swap = int(len(easy_pool) * mix_ratio)
    
    # Randomly select indices to swap
    # We want consistent results for reproducibility
    random.seed(42)
    
    # Select candidates from easy pool to become THINK (removed from nothink set)
    easy_to_think = set(random.sample([d['id'] for d in easy_pool], n_swap))
    
    # Select candidates from hard pool to become NOTHINK (added to nothink set)
    hard_to_nothink = set(random.sample([d['id'] for d in hard_pool], n_swap))
    
    # Construct final nothink indices
    # Start with all easy
    nothink_indices = set(d['id'] for d in easy_pool)
    # Remove the ones swapped to think
    nothink_indices -= easy_to_think
    # Add the ones swapped from hard
    nothink_indices |= hard_to_nothink
    
    print(f"Split Statistics:")
    print(f"  Total: {total_tasks}")
    print(f"  Alpha: {alpha}")
    print(f"  Target Nothink Count: {len(easy_pool)}")
    print(f"  Mixing Ratio: {mix_ratio}")
    print(f"  Swapped Count: {n_swap}")
    
    # Report thresholds
    if easy_pool:
        last_easy = easy_pool[-1]
        print(f"  Complexity Cutoff (Cells, LTP): {last_easy['score']}")
    
    # Generate Output
    print(f"Generating output file: {output_file}")
    
    with open(nothink_file, 'r') as f_no, \
         open(think_file, 'r') as f_think, \
         open(output_file, 'w') as f_out:
        
        for i, (line_no, line_think) in enumerate(zip(f_no, f_think)):
            if i in nothink_indices:
                f_out.write(line_no)
            else:
                f_out.write(line_think)
                
    print("Done.")

if __name__ == '__main__':
    main()
