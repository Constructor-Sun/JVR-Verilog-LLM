import pandas as pd

def main():
    file_path = './data/training_data/zhuyaoyu/CodeV-R1-dataset/codev_r1_rl_val.parquet'
    # pd.set_option('display.max_colwidth', None)

    df = pd.read_parquet(file_path)
    print(f"File: {file_path}")
    print(f"Number of examples: {len(df)}")
    print(f"One example: \n{df.iloc[0]}")

if __name__ == "__main__":
    main()
