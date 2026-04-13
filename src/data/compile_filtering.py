import re
import json
import subprocess
import os
import sys

def extract_verilog(response):
    # Remove ```verilog and ``` markers
    code = re.sub(r"<think>", "", response)
    code = re.sub(r"</think>", "", code)
    code = re.sub(r"```verilog\s*", "", code)
    code = re.sub(r"\s*```", "", code)
    return code.strip()

def main():
    nothink_path = "./data/training_data/integral/codev_r1_sft_nothink.jsonl"
    think_path = "./data/training_data/integral/codev_r1_sft_think.jsonl"
    
    nothink_out_path = "./data/training_data/integral/codev_r1_sft_nothink_filtered.jsonl"
    think_out_path = "./data/training_data/integral/codev_r1_sft_think_filtered.jsonl"
    failed_out_path = "./data/training_data/integral/codev_r1_sft_failed_prompts.jsonl"

    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    total = 0
    success = 0
    
    print(f"Processing {nothink_path} and {think_path}...")

    try:
        with open(nothink_path, "r", encoding="utf-8") as f_nothink, \
             open(think_path, "r", encoding="utf-8") as f_think, \
             open(nothink_out_path, "w", encoding="utf-8") as f_nothink_out, \
             open(think_out_path, "w", encoding="utf-8") as f_think_out, \
             open(failed_out_path, "w", encoding="utf-8") as f_failed_out:
            
            for line_nothink, line_think in zip(f_nothink, f_think):
                total += 1
                
                try:
                    data_nothink = json.loads(line_nothink)
                    data_think = json.loads(line_think)
                except json.JSONDecodeError:
                    print(f"JSON decode error at line {total}")
                    continue

                response = data_nothink.get("response", "")
                code = extract_verilog(response)
                
                if not code:
                    continue

                # Write code to a temporary file
                temp_file = "tmp/temp.v"
                temp_out = "tmp/temp"

                with open(temp_file, "w", encoding="utf-8") as tmp:
                    tmp.write(code)
                
                # Compile using iverilog
                compile_success = False
                try:
                    result = subprocess.run(
                        ["iverilog", "-g2012", "-o", temp_out, temp_file],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        compile_success = True
                    # else:
                    #     # Optional: print error for debugging
                    #     print(result)
                    #     pass
                except Exception as e:
                    print(f"Exception during compilation at line {total}:", e)
                finally:
                    # Clean up temporary files
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    if os.path.exists(temp_out):
                        os.remove(temp_out)

                if compile_success:
                    success += 1
                    f_nothink_out.write(json.dumps(data_nothink) + "\n")
                    f_think_out.write(json.dumps(data_think) + "\n")
                else:
                    f_failed_out.write(json.dumps({"prompt": data_nothink.get("prompt", "")}) + "\n")
                
                if total % 100 == 0:
                    print(f"Processed {total}, Success: {success}", end='\r')

        print(f"\nFinal: Total: {total}, Success: {success}")
        print(f"Saved to {nothink_out_path}")
        print(f"Saved to {think_out_path}")
        print(f"Saved failed prompts to {failed_out_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the input files exist.")

if __name__ == "__main__":
    main()
