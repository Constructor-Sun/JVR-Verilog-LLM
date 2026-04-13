# Intro

## function of python file

### zeta

1. check_diff_case: to find out the diff cases between think and nothink mode
2. check_zeta: to perform the influence of zeta value
3. plot_zeta: to plot the influence of zeta value
4. plot_performance_vs_time: to plot the performance of models vs time

### data

1. visualize_parquet: to visualize the train/val parquet file
2. inspect_prompt_type: to show the /think and /no_think mode for each input prompt
3. check_ratio: to show the ratio of think and nothink mode in a training data
4. cot_filtering: to filter the cot data in a training data
5. count_length: to show the number of examples in training data

### tokenizer

1. tokenizer_checkl: to show what will happen to '/think' and '/no_think' after tokenizing

### reflection

1. check_reflection: to show the impact of reflection mechanism
2. reflection_presample: use zeta to select samples for reflection
3. try_reflection: try to integrate the reflection mechanism to inference
