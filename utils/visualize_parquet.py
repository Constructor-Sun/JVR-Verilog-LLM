import pandas as pd
import pickle

def main():
    # df = pd.read_parquet("data/training_data/integral/codev_r1_rl_val_mixed.parquet")
    # df = pd.read_parquet("data/training_data/zhuyaoyu/CodeV-R1-dataset/codev_r1_rl_train.parquet")
    df = pd.read_parquet("data/training_data/integral/codev_train_think.parquet")

    idx = -1
    print(len(df))
    print(df.columns)
    print(df.iloc[idx])
    print("prompt: ", df.iloc[idx]["prompt"])
    print("type of prompt: ", type(df.iloc[idx]["prompt"]))
    # coverage = 0.0
    # for item in df["extra_info"]:
        # coverage += item["coverage"]
    # print(coverage / len(df))
    print(df["reward_model"].iloc[idx])
    print(df["extra_info"].iloc[idx])

    idx = -2
    print(len(df))
    print(df.columns)
    print(df.iloc[idx])
    print("prompt: ", df.iloc[idx]["prompt"])
    print("type of prompt: ", type(df.iloc[idx]["prompt"]))
    # coverage = 0.0
    # for item in df["extra_info"]:
        # coverage += item["coverage"]
    # print(coverage / len(df))
    print(df["reward_model"].iloc[idx])
    print(df["extra_info"].iloc[idx])

if __name__ == "__main__":
    main()
