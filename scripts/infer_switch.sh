python src/infer/qwen_switch.py \
    --cuda_device "0,1" \
    --tensor_parallel_size 2 \
    --input_file "data/inputs/hint_human_act_as_expert.jsonl" \
    --model_name "ckpts/DAPO/DAPO-Qwen3-8B-Mydesign-fixed-8roll/model_8roll_fixed_step_1200" \
    --output_file "data/outputs/model_8roll_fixed_step_switch_n10-1200.jsonl" \
    --do_sample \
    --n 10