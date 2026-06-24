set -x

export RAY_TMPDIR="/workspace/verl_exp/"
export WANDB_API_KEY=37f371d2968f35d69749ee52089583eb8e1f0cab
export WANDB_DIR="/workspace/verl_exp/"
export WANDB_MODE=online
export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="1,2"

nsr=True
nsr_delta=0.1

# Data config
train_prompt_bsz=128
max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 2))
filter_overlong_prompts=True
truncation="error"

# Algorithm config
adv_estimator=grpo
use_kl_in_reward=False

# Model config
use_remove_padding=True
enable_gradient_checkpointing=True

# Actor config
actor_optim_lr=1e-6
train_prompt_mini_bsz=32
ppo_micro_batch_size_per_gpu=4
use_kl_loss=True
kl_loss_coef=0.001
kl_loss_type=low_var_kl
entropy_coeff=0
param_offload=False
optimizer_offload=False

# Rollout config
rollout_name=vllm
log_prob_micro_batch_size_per_gpu=4
gpu_memory_utilization=0.40
n_resp_per_prompt=8
tensor_model_parallel_size=2

# Trainer config
critic_warmup=0
val_before_train=False
n_gpus_per_node=2
project_name="EXPS"
exp_name="Qwen2.5-1.5B-Math-NSR"
save_freq=30
test_freq=5
total_epochs=1
resume_mode=auto

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"/workspace/verl_exp"}
MODEL_PATH=${MODEL_PATH:-"/models/Qwen/Qwen2.5-1.5B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/math/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/math/test.parquet"}

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
    actor_rollout_ref.actor.nsr=${nsr} \
    actor_rollout_ref.actor.nsr_delta=${nsr_delta} \
    actor_rollout_ref.actor.optim.lr=${actor_optim_lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${optimizer_offload} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${param_offload} \
    trainer.critic_warmup=${critic_warmup} \
    trainer.val_before_train=${val_before_train} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=${resume_mode} \
    trainer.total_epochs=${total_epochs} $@
