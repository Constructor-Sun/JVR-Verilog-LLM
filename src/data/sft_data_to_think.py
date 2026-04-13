import json
import re
from pathlib import Path

def clean_answer_tags(text: str) -> str:
    # THe simpler version:
    # cleaned = text
    # while True:
    #     new_text = re.sub(r'<answer\s*>|</answer\s*>', '', cleaned, flags=re.IGNORECASE)
    #     if new_text == cleaned:
    #         break
    #     cleaned = new_text
    # more radical version:
    cleaned = re.sub(r'</?answer\b[^>]*>', '', text, flags=re.IGNORECASE)
    return cleaned.strip()

def add_newline_after_think(text: str) -> str:
    return re.sub(r'(<think\s*>)', r'\1\n', text, flags=re.IGNORECASE)

def process_line(line: str) -> str | None:
    try:
        obj = json.loads(line.strip())
    except json.JSONDecodeError:
        print("invalid passed:", line[:80])
        return None

    # 1. system: nothing changed
    # 2. prompt: add /think
    if "prompt" in obj and isinstance(obj["prompt"], str):
        obj["prompt"] = obj["prompt"].rstrip() + " /think"

    # 3. response: delete <answer> </answer>, add '\n' after <think>
    if "response" in obj and isinstance(obj["response"], str):
        obj["response"] = clean_answer_tags(obj["response"])
        obj["response"] = add_newline_after_think(obj["response"])

    return json.dumps(obj, ensure_ascii=False) + "\n"

def main():
    input_path = "./data/training_data/integral/codev_r1_sft.jsonl"
    output_path = "./data/training_data/integral/codev_r1_sft_think.jsonl"
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
    print(f"  invalid:  {skipped:,}")
    print(f"output → {output_file.absolute()}")


if __name__ == "__main__":
    main()