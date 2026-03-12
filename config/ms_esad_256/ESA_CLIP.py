from torch.utils.data import DataLoader
from geoseg.datasets.msesad_dataset import *
from geoseg.models.ESA_CLIP import ESA_CLIP

# training hparam
max_epoch = 32
train_batch_size = 4
val_batch_size = 4
lr = 5e-5
weight_decay = 2.5e-4
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "ESA_CLIP"
test_weights_name = ("ESA_CLIP")    # 验证时采用的权重文件名称
log_name = '{}'.format(weights_name)
monitor = 'val_macro_f1'
monitor_mode = 'max'
save_top_k = 5
save_last = False
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None

#  define the network
net = ESA_CLIP(use_indice=False, use_adapter=True)

# define the optimizer
optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

