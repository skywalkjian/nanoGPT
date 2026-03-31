# single-GPU short pilot for GPT-2 124M + BAR on OpenWebText
# intended for early baseline-vs-BAR comparisons, not full reproduction

out_dir = 'out-gpt2-124m-pilot-bar'

wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2-124m-pilot-bar'
tensorboard_log = True
tensorboard_run_name = 'gpt2-124m-pilot-bar'

dataset = 'openwebtext'

# 4 * 512 * 8 = 16,384 tokens / optimizer step on one GPU
batch_size = 4
block_size = 512
gradient_accumulation_steps = 8

# keep the model at GPT-2 124M scale
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0

use_block_attention_residuals = True
attn_res_num_blocks = 4
attn_res_use_rmsnorm = True

# short pilot budget
max_iters = 10000
lr_decay_iters = 10000
warmup_iters = 200

eval_interval = 500
eval_iters = 40
log_interval = 10

always_save_checkpoint = False
weight_decay = 1e-1
