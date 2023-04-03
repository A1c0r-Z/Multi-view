import torch


# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dataset setting
path = "data"
train_num = 4500
test_num = 499

# tokens setting
dic_path = path+"//corel5k_dictionary.txt"

# model parameter setting

batch_size = 100
n_view = 6
d_vec = 4096   # max_length of views
mlp_out = 2048 * 6  # d_model = 2048
d_model = 2048
mlp_hidden = 2048
drop_prob = 0.1
vf_hidden = 2048
vf_head = 8
vf_layers = 4
awf_gamma = 1
cf_hidden = 2048
cf_head = 8
cf_layers = 4
alpha = 5
beta = 0.05
n_cls = 260
s_mask = None
l_mask = None


# optimizer parameter setting
init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 50
epoch = 500
# clip = 1.0
weight_decay = 5e-4
inf = float('inf')
