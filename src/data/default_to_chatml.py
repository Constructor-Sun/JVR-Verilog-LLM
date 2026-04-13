import json

def convert_to_chatml(input_file, output_file):
    chatml_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            prompt = data.get("prompt", "")
            response = data.get("response", "")
            
            item = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
            }
            chatml_data.append(item)

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in chatml_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Saved at: {output_file}")
    print(f"Total: {len(chatml_data)}")

def main():
    input_path = "data/training_data/integral/codev_r1_sft_mixed_filtered.jsonl"
    output_path = "data/training_data/integral/codev_r1_sft_mixed_filtered_chatml.jsonl"
    convert_to_chatml(input_path, output_path)

if __name__ == "__main__":
    main()