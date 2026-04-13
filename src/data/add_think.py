import pandas as pd
import pickle
import numpy as np
import copy

THINK_SUFFIX = " /think"
NO_THINK_SUFFIX = " /no_think"

def modify_prompt_content(prompt, suffix):
    prompt_list = prompt.tolist()[1:]
    new_prompt = []
    for item in prompt_list:
        if isinstance(item, dict) and 'content' in item:
            # if item.get('role') == 'system':
            #     continue
            new_item = item.copy()
            new_item['content'] = new_item['content'] + suffix
            new_prompt.append(new_item)
        else:
            new_prompt.append(item)
    # return np.array(new_prompt, dtype=object)
    return new_prompt

def modify_extra_info(extra_info, suffix):
    mode = "think" if "/think" in suffix else "no_think"
    new_extra_info = extra_info.copy()  # 或者 copy.deepcopy(extra_info) 如果 extra_info 包含嵌套对象
    new_extra_info["mode"] = mode
    return new_extra_info

def main():
    input_file = "data/training_data/zhuyaoyu/CodeV-R1-dataset/codev_r1_rl_val.parquet"
    output_file = "data/training_data/integral/codev_val_think.parquet"
    
    df = pd.read_parquet(input_file)
    print(f"len: {len(df)}")
    print(f"proprieties: {df.columns.tolist()}")
    
    rows = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        prompt = row["prompt"]
        think_prompt = modify_prompt_content(prompt, THINK_SUFFIX)
        no_think_prompt = modify_prompt_content(prompt, NO_THINK_SUFFIX)
        think_row = copy.deepcopy(row)
        think_row["prompt"] = think_prompt
        think_row["extra_info"] = modify_extra_info(think_row["extra_info"], THINK_SUFFIX)
        print("think extra: ", think_row["extra_info"])

        no_think_row = copy.deepcopy(row)
        no_think_row["prompt"] = no_think_prompt
        no_think_row["extra_info"] = modify_extra_info(no_think_row["extra_info"], NO_THINK_SUFFIX)
        print("nothink extra: ", no_think_row["extra_info"])
        rows.append(think_row)
        rows.append(no_think_row)
    
    df_out = pd.DataFrame(rows)
    print(f"len after duplication: {len(df_out)}")
    df_out.to_parquet(output_file, index=False)
    print(f"saved at: {output_file}")


if __name__ == "__main__":
    main()
