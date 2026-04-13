import json
import pandas as pd

def jsonl_to_parquet(jsonl_path, parquet_path):
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            # Remove 'system' column if it exists
            record.pop('system', None)
            records.append(record)
    
    df = pd.DataFrame(records)
    df.to_parquet(parquet_path, index=False)

def main():
    train_file = 'data/training_data/integral/codev_r1_sft_mixed_filtered_chatml.jsonl'
    # val_file = 'data/training_data/integral/codev_r1_sft_mixed_filtered_val.jsonl'
    
    train_parquet = train_file.replace('.jsonl', '.parquet')
    jsonl_to_parquet(train_file, train_parquet)
    
    # val_parquet = val_file.replace('.jsonl', '.parquet')
    # jsonl_to_parquet(val_file, val_parquet)

if __name__ == "__main__":
    main()