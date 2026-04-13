# Order

## SFT

1. sft_data_to_think.py: convert deepseek-R1's response to think format
2. sft_data_to_nothink.py: convert deepseek-R1's response to nothink format
3. compile_filtering.py: use iverilog to filter out syntax-errorous examples
4. yosys_filter.py: use yosys to analyze complexity and mix no-think and think examples
5. default_to_chatml.py: convert format to chatml for swift SFT.
6. jsonl_to_parquet.py: convert jsonl files to parquet files.

## RL

1. get_rl_dataset.py: construct correct RL format for VERL.
2. get_mixed_rl_dataset.py: mix think and nothink examples.
3. fix_rl_dataset.py: fix the format of rl dataset.

## Data Distillation

- distill_new_data.py: distill new data from other SoTA's response.

## Others

- sft_data_split.py: split the dataset into train, val, test sets. Since GPU memory is limited, here I abort validation. But validation data can be directly opened to see if data format is correct.
- add_think.py: add think tag to the dataset.
