import json
import random
import os

def main():
    random.seed(42)

    rate = 0.02
    # New input file requested by user
    input_file = "./data/training_data/integral/codev_r1_sft_mixed_filtered_chatml.jsonl"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # Derive output names
    # input: .../codev_r1_sft_mixed_filtered.jsonl
    # output: .../codev_r1_sft_mixed_filtered_train.jsonl
    filename_no_ext = input_file.split("/")[-1].replace(".jsonl", "")
    output_dir = "./data/training_data/integral"
    
    os.makedirs(output_dir, exist_ok=True)

    data = []
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    total_len = len(data)
    print(f"Total examples: {total_len}")
    
    indices = list(range(total_len))
    random.shuffle(indices)
    
    val_size = int(total_len * rate)
    
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    
    print(f"Train size: {len(train_data)}")
    print(f"Val size: {len(val_data)}")
    
    train_out_path = os.path.join(output_dir, filename_no_ext + "_train.jsonl")
    val_out_path = os.path.join(output_dir, filename_no_ext + "_val.jsonl")
    
    print(f"Writing train set to {train_out_path}...")
    with open(train_out_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Writing val set to {val_out_path}...")
    with open(val_out_path, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("Done.")

if __name__ == "__main__":
    main()
