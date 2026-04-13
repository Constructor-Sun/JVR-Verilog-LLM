

import re
import os
import subprocess
import tempfile

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    # Check for required tags
    # print("solution_str: ", solution_str)
    # print("solution_str type: ", type(solution_str))
    # print("ground_truth: ", ground_truth)
    # print("ground_truth type: ", type(ground_truth))
    # print("extra_info: ", extra_info)
    # print("extra_info type: ", type(extra_info))

    required_tags = ["<think>", "</think>", "```verilog", "```"]
    if not all(tag in solution_str for tag in required_tags):
        return -1

    # Check for think block constraint
    if extra_info and not extra_info.get("if_think", True):
        think_pattern = r"<think>(.*?)</think>"
        match = re.search(think_pattern, solution_str, re.DOTALL)
        if match:
            think_content = match.group(1)
            if any(c not in ['\n', ' '] for c in think_content):
                return -1

    # Extract code from ```verilog block
    pattern = r"```verilog\s*(.*?)\s*```"
    match = re.search(pattern, solution_str, re.DOTALL)
    if not match:
        return -1 # Should be covered by above check, but just in case
    
    solution_code = match.group(1)
    testbench = extra_info["testbench"]
    
    # Create a temporary directory for compilation and execution
    with tempfile.TemporaryDirectory() as temp_dir:
        # File paths
        tb_file = os.path.join(temp_dir, "tb.v")
        gt_file = os.path.join(temp_dir, "gt.v")
        sol_file = os.path.join(temp_dir, "sol.v")
        gt_out = os.path.join(temp_dir, "gt_out")
        sol_out = os.path.join(temp_dir, "sol_out")
        
        # Write files
        with open(tb_file, "w") as f:
            f.write(testbench)
        with open(gt_file, "w") as f:
            f.write(ground_truth)
        with open(sol_file, "w") as f:
            f.write(solution_code)
            
        try:
            # Compile and run Ground Truth
            # iverilog -g2012 -o gt_out tb.v gt.v
            cmd_compile_gt = ["iverilog", "-g2012", "-o", gt_out, tb_file, gt_file]
            subprocess.run(cmd_compile_gt, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            
            # vvp gt_out
            cmd_run_gt = ["vvp", gt_out]
            res_gt = subprocess.run(cmd_run_gt, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
            output_gt = res_gt.stdout
            
            # Compile and run Solution
            # iverilog -g2012 -o sol_out tb.v sol.v
            cmd_compile_sol = ["iverilog", "-g2012", "-o", sol_out, tb_file, sol_file]
            subprocess.run(cmd_compile_sol, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            
            # vvp sol_out
            cmd_run_sol = ["vvp", sol_out]
            res_sol = subprocess.run(cmd_run_sol, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
            output_sol = res_sol.stdout
            
            # Compare outputs
            # Split outputs into lines and compute match ratio
            gt_lines = output_gt.splitlines()
            sol_lines = output_sol.splitlines()
            if len(gt_lines) != len(sol_lines):
                return -0.5
            matches = sum(1 for g, s in zip(gt_lines, sol_lines) if g == s)
            match_ratio = matches / len(gt_lines) if gt_lines else 1.0
            if match_ratio >= 0.99:
                return 1.0
            else:
                return -0.5
                
        except subprocess.TimeoutExpired:
            # Execution timed out
            return -1.0
        except subprocess.CalledProcessError:
            # Compilation or execution failed
            return -1.0
        except Exception as e:
            # Other errors
            return -1.0
