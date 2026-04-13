python src/infer/qwen.py \
    --cuda_device "0" \
    --tensor_parallel_size 1 \
    --model_name "ckpts/DAPO/DAPO-Qwen3-8B-Mydesign-fixed-8roll/model_8roll_fixed_step_1200" \
    --input_file "data/inputs/hint_cc.jsonl" \
    --max_tokens 16384 \
    --output_file "data/v2_cc_outputs/model_8roll_fixed_step_n1-1200.jsonl" \
    --think \
    --do_sample \
    --n 1

    # --model_name "ckpts/DAPO/DAPO-Qwen3-8B-Mydesign-fixed-8roll/model_8roll_fixed_step_1200" \
    # --output_file "data/outputs/qwen-8b-model_8roll_fixed_step_n10-1150.jsonl" \