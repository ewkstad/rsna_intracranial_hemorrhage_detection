
# TODO: Combine config_df and global params

import torch

fold_num = 0
is_vanilla = False # Determines whether or not to include final BN in pretrained network.
batch_size = 32
dimY = 512
dimX = 512
num_classes = 6
quicksleep = .05
max_sleep = .2
processed_in_batch = 0
batch_dim = (batch_size, 3, dimY, dimX)
device = torch.cuda.current_device()
batchX = torch.zeros(batch_dim).to(device)
batchY = torch.zeros((batch_size, num_classes)).to(device)
exp = 'VanillaDenseNet121Fold{}'.format(fold_num)
ROOT_DIR = "C:/Users/evbruh/Downloads/rsna-intracranial-hemorrhage-detection/"
TRAIN_DIR = ROOT_DIR + 'stage_1_train_images'
TEST_DIR = ROOT_DIR + 'stage_1_test_images'
ALL_DF_PATH = ROOT_DIR + 'all_df.p'
TRAIN_DF_PATH = ROOT_DIR + 'train_df.p'
VAL_DF_PATH = ROOT_DIR + 'val_df.p'
LOGGING_PATH = ROOT_DIR + '{}_log_dict.p'.format(exp)
