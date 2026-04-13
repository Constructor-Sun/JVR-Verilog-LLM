import os
import json
import re
import argparse
import subprocess
import tempfile
import hashlib
import random
import shutil
import math
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Import EDA tools from the project
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rl.verify import eda_tools

def parse_arguments():
    parser = argparse.ArgumentParser(description="No-think inference and think reflection for Verilog generation.")
    parser.add_argument("--model_name", type=str, default="ckpts/DAPO/DAPO-Qwen3-8B-Mydesign-fixed-8roll/model_8roll_fixed_step_1200", help="Model path")
    parser.add_argument("--cuda_device", type=str, default="0", help="CUDA device")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization ratio (0.0~1.0)")
    parser.add_argument("--input_file", type=str, default="data/inputs/hint_human_act_as_expert.jsonl", help="Input dataset")
    parser.add_argument("--output_file", type=str, default="data/outputs/reflection_wo_seq_results.jsonl", help="Output results")
    parser.add_argument("--triggered_tasks_file", type=str, default="data/outputs/triggered_tasks.json", help="Triggered tasks file")
    parser.add_argument("--max_tokens", type=int, default=1024*16, help="Max tokens")
    parser.add_argument("--num_test_vectors", type=int, default=100, help="Number of test vectors for simulation")
    parser.add_argument("--n", type=int, default=1, help="Number of completions per prompt")
    return parser.parse_args()

def extract_port_info(verilog_code):
    """
    Extract port information from Verilog code.
    """
    # Remove comments
    code = re.sub(r"//.*|/\*[\s\S]*?\*/", "", verilog_code)
    
    input_port_width = []
    output_port_width = []
    clock_ports = []
    reset_ports = []
    
    # regex for ports: input/output [msb:lsb] name
    # Handle optional logic/reg keywords and multi-line definitions
    port_pattern = r"(input|output)\s+(?:(?:reg|wire|logic)\s+)?(?:\[(\d+):(\d+)\]\s+)?(\w+)"
    matches = re.findall(port_pattern, code, re.MULTILINE)
    
    for direction, msb, lsb, name in matches:
        width = 1
        if msb and lsb:
            width = abs(int(msb) - int(lsb)) + 1
        
        if direction == "input":
            input_port_width.append((name, width))
            if re.search(r"clk|clock", name, re.I):
                clock_ports.append((name, 1))
            if re.search(r"rst|reset", name, re.I):
                reset_ports.append((name, 1, True))
        else:
            output_port_width.append((name, width))
            
    return input_port_width, output_port_width, clock_ports, reset_ports

def get_simulation_io(verilog_code, num_test_vectors=100):
    """
    Run simulation and return input/output sequences, limited to num_test_vectors.
    """
    v = eda_tools(quiet=True, random_seq_num=1, random_seq_steps=num_test_vectors)
    try:
        top_module = v.auto_top(verilog_code)
    except:
        return None, "Module parsing failed"

    port_info = extract_port_info(verilog_code)
    in_w, out_w, clk_p, rst_p = port_info
    
    uid = hashlib.md5(verilog_code.encode()).hexdigest()
    test_path = f"./tmp/simulation_{uid}"
    os.makedirs(test_path, exist_ok=True)
    
    try:
        # Use eda_tools.generate_testbench directly
        tb_module_code = v.generate_testbench(
            input_port_width=in_w,
            output_port_width=out_w,
            clock_port_polarity=clk_p,
            reset_port_polarity_sync=rst_p,
            golden_top=top_module,
            gate_top=top_module
        )
        
        # Add $monitor to the testbench
        monitor_ports = ["$time"]
        for p, _ in in_w:
            monitor_ports.append(f"{p}_in")
        for p, _ in out_w:
            monitor_ports.append(f"{p}_gate")
        
        fmt = " ".join(["%h"] * (len(monitor_ports) - 1))
        monitor_code = f'\n    initial $monitor("%t {fmt}", {", ".join(monitor_ports)});\n'
        tb_module_code = tb_module_code.replace("endmodule", monitor_code + "endmodule")
        
        # Process Verilog to add suffixes (to match TB instantiation)
        renamed_dut_code = v.process_verilog(verilog_code, v.gate_suffix)
        renamed_gold_code = v.process_verilog(verilog_code, v.golden_suffix)
        
        # Write ONLY renamed modules and TB to a SINGLE file to avoid redeclaration
        tb_path = os.path.join(test_path, "tb.v")
        with open(tb_path, "w") as f:
            f.write(renamed_gold_code + "\n")
            f.write(renamed_dut_code + "\n")
            f.write(tb_module_code)
        
        vvp_path = os.path.join(test_path, "tb.vvp")
        # Compile ONLY tb.v
        compile_cmd = f"iverilog -g2012 -o {vvp_path} -s testbench {tb_path}"
        run_cmd = f"vvp {vvp_path} +seed=0 +outerLoopNum=1 +innerLoopNum={num_test_vectors}"
        
        compile_proc = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
        if compile_proc.returncode != 0:
            return None, f"Icarus Verilog compilation failed:\n{compile_proc.stderr}"

        sim_proc = subprocess.run(run_cmd, shell=True, capture_output=True, text=True)
        if sim_proc.returncode != 0:
            return None, f"VVP simulation failed with return code {sim_proc.returncode}:\n{sim_proc.stderr}"
        
        # Only treat stderr as a failure if it contains "error" (case-insensitive).
        # Warnings or other messages are allowed.
        if sim_proc.stderr and re.search(r"error", sim_proc.stderr, re.I):
            return None, f"VVP simulation reported errors:\n{sim_proc.stderr}"

        # Capture the $monitor output from stdout
        lines = sim_proc.stdout.splitlines()
        io_lines = [l for l in lines if l.strip() and l.strip()[0].isdigit()]
        
        header = "Time " + " ".join([p for p, _ in in_w]) + " " + " ".join([p for p, _ in out_w])
        return header + "\n" + "\n".join(io_lines), None
    except Exception as e:
        return None, f"An unexpected error occurred during simulation: {str(e)}"
    # finally:
        # shutil.rmtree(test_path, ignore_errors=True)

def post_process_verilog(completion: str):
    """
    Extract clean Verilog code from model response.
    """
    pattern = r"```(?:verilog|Verilog)\s*([\s\S]*?)\s*```"
    match = re.search(pattern, completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    return completion.strip()

def extract_module_body(verilog_code: str):
    """
    Remove module head and everything after endmodule, matching logic in qwen.py.
    """
    # Remove module definition line (handling multi-line)
    # This regex looks for 'module' followed by name, optional params, optional port list, ending with ';'
    pattern_module_def = r"^\s*module\s+\w+\s*(?:#\s*\([^)]*\)\s*)?(?:\([^)]*\)\s*)?;\s*\n?"
    # Since the above regex might still be brittle for complex multi-line definitions, 
    # we can try to find the first ';' that follows 'module'
    
    lines = verilog_code.split('\n')
    start_index = 0
    module_found = False
    for i, line in enumerate(lines):
        if re.search(r"^\s*module\s+\w+", line):
            module_found = True
            # Find the line with the semicolon closing the module definition
            for j in range(i, len(lines)):
                if ';' in lines[j]:
                    start_index = j + 1
                    break
            break
    
    if module_found:
        lines = lines[start_index:]
    
    # Remove everything after endmodule
    for i, line in enumerate(lines):
        if 'endmodule' in line:
            # Keep endmodule but remove anything after it on the same line
            lines[i] = re.sub(r'endmodule\s*.*', 'endmodule', line)
            # Remove all lines after endmodule
            lines = lines[:i+1]
            break
            
    return '\n'.join(lines).strip()

def main():
    args = parse_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=20480
    )
    
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.7)
    # sampling_params = SamplingParams(
    #     max_tokens=80, # Generate at least 80 tokens
    #     temperature=0.7,
    #     top_p=getattr(args, 'top_p', 0.8),
    #     top_k=getattr(args, 'top_k', 20),
    #     logprobs=20, # Request top 100 logprobs per token
    #     skip_special_tokens=False
    # )
    # SamplingParams for no-think inference support n > 1
    sampling_params_nothink = SamplingParams(max_tokens=args.max_tokens, temperature=0.7, n=args.n)
    # sampling_params_nothink = SamplingParams(
    #     max_tokens=80, # Generate at least 80 tokens
    #     temperature=0.7,
    #     top_p=getattr(args, 'top_p', 0.8),
    #     top_k=getattr(args, 'top_k', 20),
    #     logprobs=20, # Request top 100 logprobs per token
    #     skip_special_tokens=False,
    #     n=args.n
    # )
    
    # Load triggered tasks
    triggered_tasks_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", args.triggered_tasks_file))
    with open(triggered_tasks_path, "r") as f_tasks:
        triggered_tasks_data = json.load(f_tasks)
    
    target_task_ids = set(triggered_tasks_data.get("think passed", []) + triggered_tasks_data.get("think failed", []))
    print(f"Loaded {len(target_task_ids)} target task IDs: {target_task_ids}")

    input_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", args.input_file))
    output_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", args.output_file))
    
    with open(input_file_path, "r") as f_in, open(output_file_path, "w") as f_out:
        reflection_num = 0
        for line in tqdm(f_in):
            data = json.loads(line)
            task_id = data["task_id"]
            
            if task_id not in target_task_ids:
                continue
                
            hint = data["hint"]
            
            # 1. No-think Inference
            prompt_nothink = f"### Hint: Your generated Verilog code should be in ```verilog``` format.\n{hint} /think"
            messages = [{"role": "user", "content": prompt_nothink}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            outputs = model.generate([input_text], sampling_params_nothink)
            
            # Process each of the n completions
            for output_idx in range(args.n):
                generated_response = outputs[0].outputs[output_idx].text
                
                verilog_code = post_process_verilog(generated_response)
                
                if not verilog_code:
                    continue
                    
                # 2. Reflection logic to get completion
                reflection_num += 1
                io_sequence, error = get_simulation_io(verilog_code, num_test_vectors=100)
                if error:
                    io_sequence = f"Simulation failed: {error}"
                
                # Truncate io_sequence if it's too long (avoid prompt length exceeding max_model_len)
                if io_sequence and len(io_sequence) > 10000:
                    # Keep the first 10,000 characters and end at a newline to avoid cutting a line in half
                    truncated_io = io_sequence[:10000]
                    last_newline = truncated_io.rfind("\n")
                    if last_newline != -1:
                        io_sequence = truncated_io[:last_newline] + "\n... (truncated due to length)"
                    else:
                        io_sequence = truncated_io + "\n... (truncated due to length)"
                
                # 3. Think Reflection
                reflection_prompt = (
                    f"Original Prompt: {hint}\n\n"
                    f"Generated Code:\n```verilog\n{verilog_code}\n```\n\n"
                    # f"Simulation I/O Sequences (Time Input_Values Output_Values):\n{io_sequence}\n\n"
                    # f"Task: Analyze the provided I/O sequences to determine if the generated Verilog code functions correctly according to the original prompt requirements.\n\n"
                    f"Task: Determine if the generated Verilog code functions correctly according to the original prompt requirements.\n\n"
                    f"Important Considerations:\n"
                    # f"1. Initial 'x' or 'z' values: At the very beginning of the simulation (Time 0 or during the initial reset period), it is normal for signals to have 'x' or 'z' values. These should be ignored if the circuit transitions to a correct state after the reset is complete.\n"
                    f"2. Functional Accuracy: Prioritize functional behavior over stylistic choices. If the circuit's logic matches the requirements, it is considered correct.\n"
                    f"3. Verification Result:\n"
                    f"   - If the code is already correct, state that it is correct and provide the original code in a ```verilog``` block.\n"
                    f"   - If there are functional errors, identify the mistakes clearly and provide the corrected Verilog code in a ```verilog``` block.\n\n"
                    f"Your response MUST always include the Verilog code (original or corrected) inside a ```verilog``` block. /think"
                )
                
                messages_reflect = [{"role": "user", "content": reflection_prompt}]
                reflect_text = tokenizer.apply_chat_template(messages_reflect, tokenize=False, add_generation_prompt=True)
                
                outputs_reflect = model.generate([reflect_text], sampling_params)
                reflection_response = outputs_reflect[0].outputs[0].text
                completion_code = post_process_verilog(reflection_response)
                completion_body = extract_module_body(completion_code)
                
                result = {
                    "task_id": task_id,
                    "completion": completion_body,
                    "nothink_response": generated_response,
                    "io_sequence": io_sequence
                }
                f_out.write(json.dumps(result) + "\n")
        print('reflection_num: ', reflection_num)

if __name__ == "__main__":
    main()
