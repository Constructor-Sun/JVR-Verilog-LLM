#!/bin/bash
# export HYDRA_FULL_ERROR=1 
export VLLM_USE_V1=1

set -x
set -euxo pipefail

ORI_PWD=${PWD}
[ -d "verl" ] && cd verl || echo "verl sub-directory not found! Or you are already in verl directory."

project_name='DAPO'
exp_name='DAPO-Qwen3-8B-Mydesign-fixed-8roll'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 2))
overlong_penalty_factor=0.05

loss_agg_mode="token-mean"

# An early version for DAPO
enable_filter_groups=True
train_prompt_bsz=32
train_prompt_mini_bsz=8
n_resp_per_prompt=8
use_token_level_loss=True

# mixed thinking and nothinking
mixed_thinking=True
dapo_tau=0.25 # threshold of average score per example to enable pair-wise reward, current 0.5*train_prompt_mini_bsz*0.5
dapo_gamma=0.5 # improved factor
expected_reward=0.5
log_freq=1

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-16}
# Paths
# Algorithm
## Train
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 16))
## Validation
val_top_k=-1 # 0 for HF rollout, -1 for vLLM rollout


# Performance Related Parameter
USER_GPUS_PER_NODE=2
SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
sp_size=2
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
offload=True
gen_tp=1
ppo_max_token_len_per_gpu=18432
num_gpu=$(($USER_GPUS_PER_NODE * $SLURM_JOB_NUM_NODES))

RAY_DATA_HOME=${RAY_DATA_HOME:-"${ORI_PWD}"}
# MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/sft-8b/v0-20260127-202025/checkpoint-1100"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/ckpts/DAPO/DAPO-Qwen3-8B-Codev/model_step_150"}
SAVE_DIR=${SAVE_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/training_data/integral/codev_train_think.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/training_data/integral/codev_val_think.parquet"}

# Train over a single node, 2 A100-80GB GPUs.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.mixed_thinking=${mixed_thinking} \
    +data.gen_batch_size=$((($train_prompt_bsz + $num_gpu - 1) / $num_gpu * $num_gpu)) \
    data.train_batch_size=${train_prompt_bsz} \
    data.truncation='left' \
    data.val_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=8 \
    +algorithm.filter_groups.enable=${enable_filter_groups} \
    +algorithm.filter_groups.max_num_gen_batches=999 \
    +algorithm.filter_groups.metric=acc \
    +algorithm.filter_groups.accelerate=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}\
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=0.5 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    reward_model.reward_manager=dapo \
    reward_model.mixed_thinking=${mixed_thinking} \
    reward_model.dapo_gamma=${dapo_gamma} \
    reward_model.dapo_tau=${dapo_tau} \
    reward_model.expected_reward=${expected_reward} \
    reward_model.log_freq=${log_freq} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.fsdp_config.model_dtype=bf16 \
    +custom_reward_function.overlong_buffer.enable=${enable_overlong_buffer} \
    +custom_reward_function.overlong_buffer.len=${overlong_buffer_len} \
    +custom_reward_function.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    custom_reward_function.path=${ORI_PWD}/src/rl/codev.py \
    custom_reward_function.name=compute_pair_score_wrapper \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='qwen3-8b-rl' \
    trainer.experiment_name='qwen3-4.8kdata-mixed' \
    trainer.n_gpus_per_node=$USER_GPUS_PER_NODE \
    trainer.nnodes=$SLURM_JOB_NUM_NODES \
    trainer.val_before_train=False \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.resume_mode=auto \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=8
