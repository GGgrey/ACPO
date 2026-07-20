#!/usr/bin/env bash
set -x

# Environment
export WORKING_DIR="${WORKING_DIR:-${PWD}}"
export RAY_TMPDIR="${WORKING_DIR}"
export WANDB_API_KEY=37f371d2968f35d69749ee52089583eb8e1f0cab
export WANDB_DIR="${WORKING_DIR}"
export WANDB_MODE=online
export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="1,2,3,4"

# Data config
train_prompt_bsz=256
max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 2))
filter_overlong_prompts=False
truncation='left'

# Algorithm config
adv_estimator=grpo
use_kl_in_reward=False

# Model config
use_remove_padding=True
enable_gradient_checkpointing=True

# Actor config
optim_lr=1e-6
train_prompt_mini_bsz=8
ppo_micro_batch_size_per_gpu=1
use_kl_loss=True
kl_loss_coef=0.001
kl_loss_type=low_var_kl
entropy_coeff=0
actor_param_offload=False
optimizer_offload=False

# Rollout config
log_prob_micro_batch_size_per_gpu=1
tensor_model_parallel_size=2
rollout_name=vllm
gpu_memory_utilization=0.30
n_resp_per_prompt=8
train_temperature=1.0
train_top_p=1.0
train_top_k=-1
val_temperature=1.0
val_top_p=0.95
val_top_k=-1
val_do_sample=True
val_n=1

# Reference config
ref_param_offload=False

# Trainer config
critic_warmup=0
val_before_train=False
logger='["console","wandb"]'
project_name="EXPS"
exp_name="GRPO-Qwen2.5-7B-Instruct"
n_gpus_per_node=4
save_freq=20
test_freq=5
resume_mode=auto
total_epochs=1

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${WORKING_DIR}"}
MODEL_PATH=${MODEL_PATH:-"/models/Qwen/Qwen2.5-7B-Instruct"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo/dapo-math-17k-processed.parquet"}
TEST_FILE=${TEST_FILE:-"['${RAY_DATA_HOME}/data/aime24/aime24.parquet','${RAY_DATA_HOME}/data/aime25/aime25.parquet','${RAY_DATA_HOME}/data/amc23/amc23.parquet','${RAY_DATA_HOME}/data/math500/math500.parquet','${RAY_DATA_HOME}/data/gaokao2023en/gaokao2023en.parquet','${RAY_DATA_HOME}/data/olympiadbench/olympiadbench.parquet','${RAY_DATA_HOME}/data/minerva/minerva.parquet']"}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-"${RAY_DATA_HOME}/data/rollout_data/${project_name}/${exp_name}"}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-"${RAY_DATA_HOME}/data/validation_data/${project_name}/${exp_name}"}

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=${filter_overlong_prompts} \
    data.truncation=${truncation} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.model.enable_gradient_checkpointing=${enable_gradient_checkpointing} \
    actor_rollout_ref.actor.optim.lr=${optim_lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${optimizer_offload} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.temperature=${train_temperature} \
    actor_rollout_ref.rollout.top_p=${train_top_p} \
    actor_rollout_ref.rollout.top_k="${train_top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k="${val_top_k}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample} \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_param_offload} \
    trainer.critic_warmup=${critic_warmup} \
    trainer.val_before_train=${val_before_train} \
    trainer.logger="${logger}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.rollout_data_dir=${ROLLOUT_DATA_DIR} \
    trainer.validation_data_dir=${VALIDATION_DATA_DIR} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=${resume_mode} \
    trainer.total_epochs=${total_epochs} $@
