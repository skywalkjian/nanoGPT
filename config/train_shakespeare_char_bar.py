# train a miniature character-level shakespeare model with Block Attention Residuals

out_dir = 'out-shakespeare-char-bar'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt-bar'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

use_block_attention_residuals = True
attn_res_num_blocks = 3
attn_res_use_rmsnorm = True

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100
