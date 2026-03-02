#!/bin/bash
set -euo pipefail

# ray job submit --address="http://0.0.0.0:8265" \
#     --runtime-env=verl/trainer/runtime_env.yaml \
#     --no-wait \
#     -- \


train_files="${TRAIN_FILES:-['/path/to/train.parquet']}"
test_files="${VAL_FILES:-['/path/to/val.parquet']}"
infer_ppo_max_token_len=69632
train_ppo_max_token_len=22528
loss_agg_mode="token-mean"
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
ckpt_save_path="${CKPT_SAVE_PATH:-./ckpts/longrlvr_example}"
model_path="${MODEL_PATH:-"/path/to/base-model"}"

clip_ratio_low=0.2
clip_ratio_high=0.28

resume_mode="auto"

python3 -m recipe.dapo.main_dapo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=32 \
    data.max_prompt_length=65536 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=False \
    data.filter_overlong_prompts_workers=16 \
    data.truncation='left' \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${train_ppo_max_token_len} \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.disable_log_stats=True \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    reward_model.reward_manager=longrl_async \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.max_num_gen_batches=10 \
    algorithm.filter_groups.metric=seq_final_reward \
    trainer.logger=['console,wandb'] \
    trainer.project_name='verl_dapo_longrl' \
    trainer.experiment_name='qwen3_4b_instruct_longrl_all_recall' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_local_dir=${ckpt_save_path} \
    trainer.resume_mode=${resume_mode} \
    trainer.resume_from_path=${ckpt_save_path} \
    trainer.total_epochs=5
