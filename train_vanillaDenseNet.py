
import torch
import pandas as pd
import torch.nn as nn
from nets import DenseNet121Vanilla
from preprocessing_utils import DICOMPreprocessor, Controller, DataLoader
from config_dc import config_dc
from lr_scheduler import ReduceLROnPlateau
from train_eval_utils import train_model
from global_params import fold_num, TRAIN_DF_PATH, VAL_DF_PATH


args = config_dc


model = DenseNet121Vanilla()


params_to_update = ['features.10.denselayer1.norm1.weight'
                    'features.10.denselayer1.norm1.bias',
                    'features.10.denselayer1.conv1.weight',
                    'features.10.denselayer1.norm2.weight',
                    'features.10.denselayer1.norm2.bias',
                    'features.10.denselayer1.conv2.weight',
                    'features.10.denselayer2.norm1.weight',
                    'features.10.denselayer2.norm1.bias',
                    'features.10.denselayer2.conv1.weight',
                    'features.10.denselayer2.norm2.weight',
                    'features.10.denselayer2.norm2.bias',
                    'features.10.denselayer2.conv2.weight',
                    'features.10.denselayer3.norm1.weight',
                    'features.10.denselayer3.norm1.bias',
                    'features.10.denselayer3.conv1.weight',
                    'features.10.denselayer3.norm2.weight',
                    'features.10.denselayer3.norm2.bias',
                    'features.10.denselayer3.conv2.weight',
                    'features.10.denselayer4.norm1.weight',
                    'features.10.denselayer4.norm1.bias',
                    'features.10.denselayer4.conv1.weight',
                    'features.10.denselayer4.norm2.weight',
                    'features.10.denselayer4.norm2.bias',
                    'features.10.denselayer4.conv2.weight',
                    'features.10.denselayer5.norm1.weight',
                    'features.10.denselayer5.norm1.bias',
                    'features.10.denselayer5.conv1.weight',
                    'features.10.denselayer5.norm2.weight',
                    'features.10.denselayer5.norm2.bias',
                    'features.10.denselayer5.conv2.weight',
                    'features.10.denselayer6.norm1.weight',
                    'features.10.denselayer6.norm1.bias',
                    'features.10.denselayer6.conv1.weight',
                    'features.10.denselayer6.norm2.weight',
                    'features.10.denselayer6.norm2.bias',
                    'features.10.denselayer6.conv2.weight',
                    'features.10.denselayer7.norm1.weight',
                    'features.10.denselayer7.norm1.bias',
                    'features.10.denselayer7.conv1.weight',
                    'features.10.denselayer7.norm2.weight',
                    'features.10.denselayer7.norm2.bias',
                    'features.10.denselayer7.conv2.weight',
                    'features.10.denselayer8.norm1.weight',
                    'features.10.denselayer8.norm1.bias',
                    'features.10.denselayer8.conv1.weight',
                    'features.10.denselayer8.norm2.weight',
                    'features.10.denselayer8.norm2.bias',
                    'features.10.denselayer8.conv2.weight',
                    'features.10.denselayer9.norm1.weight',
                    'features.10.denselayer9.norm1.bias',
                    'features.10.denselayer9.conv1.weight',
                    'features.10.denselayer9.norm2.weight',
                    'features.10.denselayer9.norm2.bias',
                    'features.10.denselayer9.conv2.weight',
                    'features.10.denselayer10.norm1.weight',
                    'features.10.denselayer10.norm1.bias',
                    'features.10.denselayer10.conv1.weight',
                    'features.10.denselayer10.norm2.weight',
                    'features.10.denselayer10.norm2.bias',
                    'features.10.denselayer10.conv2.weight',
                    'features.10.denselayer11.norm1.weight',
                    'features.10.denselayer11.norm1.bias',
                    'features.10.denselayer11.conv1.weight',
                    'features.10.denselayer11.norm2.weight',
                    'features.10.denselayer11.norm2.bias',
                    'features.10.denselayer11.conv2.weight',
                    'features.10.denselayer12.norm1.weight',
                    'features.10.denselayer12.norm1.bias',
                    'features.10.denselayer12.conv1.weight',
                    'features.10.denselayer12.norm2.weight',
                    'features.10.denselayer12.norm2.bias',
                    'features.10.denselayer12.conv2.weight',
                    'features.10.denselayer13.norm1.weight',
                    'features.10.denselayer13.norm1.bias',
                    'features.10.denselayer13.conv1.weight',
                    'features.10.denselayer13.norm2.weight',
                    'features.10.denselayer13.norm2.bias',
                    'features.10.denselayer13.conv2.weight',
                    'features.10.denselayer14.norm1.weight',
                    'features.10.denselayer14.norm1.bias',
                    'features.10.denselayer14.conv1.weight',
                    'features.10.denselayer14.norm2.weight',
                    'features.10.denselayer14.norm2.bias',
                    'features.10.denselayer14.conv2.weight',
                    'features.10.denselayer15.norm1.weight',
                    'features.10.denselayer15.norm1.bias',
                    'features.10.denselayer15.conv1.weight',
                    'features.10.denselayer15.norm2.weight',
                    'features.10.denselayer15.norm2.bias',
                    'features.10.denselayer15.conv2.weight',
                    'features.10.denselayer16.norm1.weight',
                    'features.10.denselayer16.norm1.bias',
                    'features.10.denselayer16.conv1.weight',
                    'features.10.denselayer16.norm2.weight',
                    'features.10.denselayer16.norm2.bias',
                    'features.10.denselayer16.conv2.weight',
                    'bn0.weight',
                    'bn0.bias',
                    'classifier.weight',
                    'classifier.bias']

for name, param in model.named_parameters():
    if name in params_to_update:
        print(name)
        param.requires_grad = True
    else:
        param.requires_grad = False

model = model.cuda()


weights = [1, 2, 2, 2, 2, 2]
criterion = nn.BCELoss(torch.Tensor(weights).cuda())


TRAIN_DF_PATH = TRAIN_DF_PATH[:-2] + '_fold_{}'.format(fold_num) + TRAIN_DF_PATH[-2:]
VAL_DF_PATH = VAL_DF_PATH[:-2] + '_fold_{}'.format(fold_num) + VAL_DF_PATH[-2:]


# TODO: Literally out of time, training on half the dataframe
train_df = pd.read_pickle(TRAIN_DF_PATH)
train_df = train_df[:train_df.shape[0]//2]

val_df = pd.read_pickle(VAL_DF_PATH)


print('train_df shape, ', train_df.shape, ' __||__ ',  'val_df shape, ', val_df.shape)


dataset_loaders = {
    'train':DataLoader(Controller(train_df),
                       DICOMPreprocessor(augment=True)),
    'val':DataLoader(Controller(val_df),
                     DICOMPreprocessor(augment=True))
}


dataset_sizes = {
    'train':dataset_loaders['train'].shape(),
    'val':dataset_loaders['val'].shape()
}


RLRP_agent = ReduceLROnPlateau('min')


num_epochs = 5


best_model = train_model(args,
                         model,
                         criterion,
                         dataset_loaders,
                         dataset_sizes,
                         RLRP_agent,
                         num_epochs)

print(best_model)