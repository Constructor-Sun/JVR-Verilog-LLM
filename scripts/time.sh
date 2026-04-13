#!/bin/bash

python src/infer/qwen.py \
    --cuda_device "1" \
    --model_name "models/zhuyaoyu/CodeV-R1-RL-Qwen-7B" \
    --input_file "data/inputs/hint_human_act_as_expert.jsonl" \
    --output_file "data/time/codev_human_output.jsonl" \
    --do_sample \
    --think \
    --n 1 \
    --tensor_parallel_size 1

    # --model_name "models/tttboy/Verirl-CodeQwen2.5" \
    # "models/zhuyaoyu/CodeV-R1-RL-Qwen-7B"
