import json
import re
from pathlib import Path

def extract_verilog_only(response: str) -> str:
    # 1. verilog
    verilog_match = re.search(r'```verilog\s*.*?\s*```', response, re.DOTALL | re.IGNORECASE)
    if verilog_match:
        return verilog_match.group(0).strip()
    
    # 2. verilog，then <answer></answer>
    answer_match = re.search(r'<answer\b[^>]*>(.*?)</answer\b[^>]*>', response, re.DOTALL | re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()
    
    # 3. then <answer>
    answer_start_match = re.search(r'<answer\b[^>]*>(.*)', response, re.DOTALL | re.IGNORECASE)
    if answer_start_match:
        return answer_start_match.group(1).strip()
    else:
        print("response: ", response)
    
    return ""

def process_line(line: str) -> str | None:
    try:
        obj = json.loads(line.strip())
    except json.JSONDecodeError:
        return None

    if "prompt" in obj and isinstance(obj["prompt"], str):
        obj["prompt"] = obj["prompt"].rstrip() + " /no_think"

    if "response" in obj and isinstance(obj["response"], str):
        obj["response"] = "<think>\n\n</think>\n\n" + extract_verilog_only(obj["response"])

    return json.dumps(obj, ensure_ascii=False) + "\n"

def main():
    input_path = "./data/training_data/integral/codev_r1_sft.jsonl"
    output_path = "./data/training_data/integral/codev_r1_sft_nothink.jsonl"
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.is_file():
        print(f"file does not exist -> {input_file}")
        return

    total = 0
    processed = 0
    skipped = 0

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            total += 1
            new_line = process_line(line)
            if new_line is not None:
                fout.write(new_line)
                processed += 1
            else:
                skipped += 1

    print(f"finished!")
    print(f"  total:     {total:,}")
    print(f"  success:   {processed:,}")
    print(f"  invalid:   {skipped:,}")
    print(f"output → {output_file.absolute()}")

if __name__ == "__main__":
    main()