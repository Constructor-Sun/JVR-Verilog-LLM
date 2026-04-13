import pandas as pd
import numpy as np
import os
import pickle
import re
import subprocess
import tempfile
from tqdm import tqdm

def calculate_coverage(dut_code, tb_code, module_name="dut"):
    with tempfile.TemporaryDirectory() as temp_dir:
        dut_path = os.path.join(temp_dir, f"{module_name}.v")
        tb_path = os.path.join(temp_dir, "testbench.v")
        
        with open(dut_path, "w") as f:
            f.write(dut_code)
        with open(tb_path, "w") as f:
            f.write(tb_code)
            
        # Compile with Verilator
        # Added -Wno-fatal to avoid failing on warnings like UNUSEDSIGNAL
        # Added --timing to support delays in testbench
        cmd = [
            "verilator",
            "--cc", "--exe", "--build", "--main", "--coverage", "-Wall", "--timing",
            "-Wno-fatal", "--top-module", "testbench",
            "--threads", "64",
            "--build-jobs", "64",
            tb_path, dut_path
        ]
        
        try:
            # Run compilation
            subprocess.run(cmd, cwd=temp_dir, capture_output=True, check=True)
            
            # Find executable
            obj_dir = os.path.join(temp_dir, "obj_dir")
            exe_path = os.path.join(obj_dir, "Vtestbench")
            if not os.path.exists(exe_path):
                # Fallback check
                if os.path.exists(os.path.join(obj_dir, "Vdut")):
                     exe_path = os.path.join(obj_dir, "Vdut")
            
            if not os.path.exists(exe_path):
                return 0.0
                
            # Run simulation
            subprocess.run([exe_path], cwd=temp_dir, capture_output=True, check=True)
            
            # Parse coverage.dat
            cov_dat_path = os.path.join(temp_dir, "coverage.dat")
            if not os.path.exists(cov_dat_path):
                return 0.0
                
            total_points = 0
            covered_points = 0
            
            with open(cov_dat_path, "r") as f:
                for line in f:
                    if line.startswith("C "):
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            count = int(parts[-1])
                            total_points += 1
                            if count > 0:
                                covered_points += 1
                                
            if total_points == 0:
                return 0.0
                
            return (covered_points / total_points) * 100.0
            
        except subprocess.CalledProcessError as e:
            # Compilation or simulation failed
            # print(f"Verilator failed: {e}")
            # if e.stderr:
            #     print(f"STDERR: {e.stderr.decode('utf-8') if isinstance(e.stderr, bytes) else e.stderr}")
            # if e.stdout:
            #     print(f"STDOUT: {e.stdout.decode('utf-8') if isinstance(e.stdout, bytes) else e.stdout}")
            # print("dut_code: ", dut_code)
            # print("tb_code: ", tb_code)
            return 0.0
        except Exception as e:
            # print(f"Coverage calculation error: {e}")
            return 0.0

def generate_testbench(module_name, input_ports, output_ports, repeat_count=100):
    """
    Generates a standardized Verilog testbench for comparing DUT vs Reference.
    Inputs/Outputs are lists of tuples: [('name', width), ...]
    """
    input_ports = list(input_ports)
    output_ports = list(output_ports)
    tb = ["`timescale 1ns/1ps\nmodule testbench;"]
    
    # 1. Signal Declarations
    for name, w in input_ports:
        tb.append(f"    reg  {f'[{w-1}:0] ' if w > 1 else ''}{name};")
    for name, w in output_ports:
        tb.append(f"    wire {f'[{w-1}:0] ' if w > 1 else ''}{name};")

    # Identification of Control Signals
    clks = [n for n, w in input_ports if re.search(r'clk|clock', n, re.I)]
    rsts = [n for n, w in input_ports if re.search(r'rst|reset', n, re.I)]
    main_clk = clks[0] if clks else "clk"
    data_inputs = [p for p in input_ports if p[0] not in clks and p[0] not in rsts]

    # 2. DUT Instantiation
    tb.append(f"\n    {module_name} dut (")
    tb.append(",\n".join([f"        .{n}({n})" for n, _ in (input_ports + output_ports)]))
    tb.append("    );")

    # 3. Clock Generation
    if clks:
        for clk in clks:
            tb.append(f"\n    initial {clk} = 0;\n    always #5 {clk} = ~{clk};")
    else:
        tb.append("\n    // Auto-generated clock (module has no clock input)")
        tb.append("    reg clk;")
        tb.append("    initial clk = 0;\n    always #5 clk = ~clk;")

    # 4. Test Stimulus Logic
    tb.append("\n    initial begin")
    # Initialize all inputs
    for n, _ in input_ports:
        if n not in clks: tb.append(f"        {n} = 0;")
    
    # Suggestion 4: Robust Multi-Reset Sequence
    if rsts:
        tb.append("\n        // Reset Phase")
        for rst in rsts: tb.append(f"        {rst} = {'0' if 'n' in rst.lower() else '1'};")
        tb.append("        #20;")
        for rst in rsts: tb.append(f"        {rst} = {'1' if 'n' in rst.lower() else '0'};")
        tb.append(f"        repeat(5) @(posedge {main_clk});")

    # Suggestion 5: Phase 1 - Directed Corner Cases
    tb.append("\n        // Phase 1: Directed Corners")
    for name, w in data_inputs:
        for val in [0, (1 << w) - 1, (1 << (w-1))]:
            tb.append(f"        @(posedge {main_clk}) {name} <= {val};")

    # Suggestion 2 & 5: Phase 2 - Scalable Random Vectors
    tb.append(f"\n        // Phase 2: Random Stress ({repeat_count} iterations)")
    tb.append(f"        repeat ({repeat_count}) begin\n            @(posedge {main_clk});")
    for name, w in data_inputs:
        chunks = (w + 31) // 32
        if chunks > 1:
            urandom_calls = ", ".join(["$urandom"] * chunks)
            val_expr = f"{w}'({{ {urandom_calls} }})"
        else:
            val_expr = f"{w}'($urandom)"
        tb.append(f"            {name} <= {val_expr};")
    tb.append("        end")
    
    tb.append(f"\n        repeat(10) @(posedge {main_clk});\n        $finish;\n    end")

    # 5. Standardized Output for Comparison
    all_names = [n for n, _ in (input_ports + output_ports)]
    fmt = " ".join(["%h"] * len(all_names))
    tb.append(f"\n    initial $monitor(\"%t {fmt}\", $time, {', '.join(all_names)});")

    tb.append("\nendmodule\n")
    return "\n".join(tb)

def handle_reward(reward_model):
    ground_truth = reward_model["ground_truth"]
    
    testbench = None
    coverage = 0.0

    # Deserialize the pickled bytes into a Python dict
    if isinstance(ground_truth, bytes):
        deserialized = pickle.loads(ground_truth)
        # Reconstruct the ground_truth dict as specified
        reward_model["ground_truth"] = deserialized["answer"]["code"]
        input_port_width = deserialized["answer"]["input_port_width"]
        output_port_width = deserialized["answer"]["output_port_width"]
        # print("input_port_width: ", input_port_width)
        # print("output_port_width: ", output_port_width)
        
        # Extract module name from ground_truth
        match = re.search(r"module\s+(\w+)", reward_model["ground_truth"])
        module_name = match.group(1) if match else "top_module"
        
        testbench = generate_testbench(
            module_name, 
            input_port_width, 
            output_port_width
        )
        
        # Calculate coverage
        coverage = calculate_coverage(
            reward_model["ground_truth"],
            testbench,
            module_name=module_name
        )
        
    style = reward_model.pop("style", None)
    
    extra_info = {
        "testbench": testbench,
        "style": style,
        "coverage": coverage
    }
    
    return reward_model, extra_info

def handle_prompt(prompt):
    # Convert numpy array to list if needed
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()

    # Exclude the first dict (system message) from the prompt list
    if isinstance(prompt, list) and len(prompt) > 1:
        prompt = prompt[1:]
    
    for item in prompt:
        if isinstance(item, dict) and item["role"] == "user":
            item["content"] += " /think"
    return prompt

def main():
    path = "data/training_data/zhuyaoyu/CodeV-R1-dataset/codev_r1_rl_train.parquet"
    output_dir = "data/training_data/integral"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, path.split("/")[-1])
    
    df = pd.read_parquet(path, columns=["data_source", "prompt", "reward_model"])
    df["prompt"] = df["prompt"].apply(handle_prompt)

    tqdm.pandas(desc="Processing reward_model")
    processed_data = df["reward_model"].progress_apply(handle_reward)
    
    df["reward_model"] = processed_data.apply(lambda x: x[0])
    df["extra_info"] = processed_data.apply(lambda x: x[1])

    # Filter out rows with 0.0 coverage
    df = df[df["extra_info"].apply(lambda x: x.get("coverage", 0.0) > 0.0)]
    
    print(f"Total number of rows: {len(df)}")
    if len(df) > 0:
        mean_coverage = df["extra_info"].apply(lambda x: x.get("coverage", 0.0)).mean()
        print(f"Mean coverage: {mean_coverage}")

    df = df[["prompt", "data_source", "reward_model", "extra_info"]]
    df.to_parquet(output_path, index=False)

if __name__ == "__main__":
    df = main()
