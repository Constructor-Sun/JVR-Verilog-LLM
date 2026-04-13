python src/infer/qwen.py \
    --cuda_device "1" \
    --tensor_parallel_size 1 \
    --input_file "data/inputs/hint_machine_act_as_expert.jsonl" \
    --model_name "ckpts/DAPO/DAPO-Qwen3-8B-Mydesign-fixed-8roll/model_8roll_fixed_step_1200" \
    --max_tokens 16384\
    --output_file "data/machine_outputs/machine_model_8roll_fixed_step_nothink_n10-1200.jsonl" \
    --do_sample \
    --n 10

    # --model_name "models/model_8roll_fixed_step_1150" \
    # --output_file "data/outputs/qwen-8b-model_8roll_fixed_step_n10-1150.jsonl" \