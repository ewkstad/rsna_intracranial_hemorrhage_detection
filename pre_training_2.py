# Credit to https://www.kaggle.com/wfwiggins203 in https://www.kaggle.com/wfwiggins203/eda-dicom-tags-windowing-head-cts

import time
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import optim
from dataframe_utils import *

import torch.nn as nn
from tqdm import tqdm
import numpy as np

from paths import *

from nets import DenseNet121Localization
from global_params import batch_size, batchX, batchY, device

from preprocessing_utils import DICOMPreprocessor, Controller, DataLoader

plt.style.use('grayscale')


all_df = pd.read_csv(ROOT_DIR + 'stage_1_train.csv')

all_df[['ID', 'Subtype']] = all_df['ID'].str.rsplit(pat='_', n=1, expand=True)


all_df['ID'] = all_df['ID'].apply(fix_id)


all_df = all_df.pivot_table(index='ID', columns='Subtype').reset_index()


all_df['Label', 'none'] = ((all_df.Label['any'].values + 1)%2)


all_df['filepath'] = all_df['ID'].apply(id_to_filepath)


all_df = all_df[[('ID',                    ''),
                 ('Label',             'none'),
                 ('Label',         'epidural'),
                 ('Label', 'intraparenchymal'),
                 ('Label', 'intraventricular'),
                 ('Label',     'subarachnoid'),
                 ('Label',         'subdural'),
                 ('filepath',              '')]]


num_classes = 6


all_df = all_df.reindex(np.random.permutation(all_df.index))


all_df.index = np.arange(all_df.shape[0]).astype('int64')


train_df = all_df[:int(.9*all_df.shape[0])]
train_df = train_df[:train_df.shape[0]-train_df.shape[0]%batch_size]

val_df = all_df[int(.9*all_df.shape[0]):]
val_df = val_df[:val_df.shape[0]-val_df.shape[0]%batch_size]


##################################
import operator
labels = list(train_df.Label.columns)
original_counts = dict(train_df.Label.sum(axis=0))
major_class, major_class_count = max(original_counts.items(), key=operator.itemgetter(1))
minor_classes_counts = {x:original_counts[x] for x in labels if x is not major_class}
minor_classes_count = sum(minor_classes_counts.values())
ratio = major_class_count/minor_classes_count
new_minor_classes_counts = {x:int(original_counts[x]*ratio) for x in minor_classes_counts}
new_counts = {**{major_class:major_class_count}, **new_minor_classes_counts}
class_dfs = {x:train_df[train_df.Label[x]==1] for x in labels}
for x in labels:
    if x is major_class:
        class_dfs[x] = class_dfs[x]
    else:
        class_dfs[x] = class_dfs[x].sample(new_counts[x], replace=True)


train_df = pd.concat([class_dfs[x] for x in labels])


train_df.index = np.arange(train_df.shape[0]).astype('int64')


train_df = train_df.reindex(np.random.permutation(train_df.index))


train_df.index = np.arange(train_df.shape[0]).astype('int64')


##################################


train_dicom_preprocessor = DICOMPreprocessor(augment=True)
val_dicom_preprocessor = DICOMPreprocessor(augment=True)


train_controller = Controller(data=train_df)
val_controller = Controller(data=val_df)


train_dataloader = DataLoader(train_controller, train_dicom_preprocessor)
val_dataloader = DataLoader(val_controller, val_dicom_preprocessor)


model = DenseNet121Localization()


params_to_update = [
    'features.10.denselayer1.norm1.weight',
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
    'classifier.1.weight',
    'classifier.1.bias',
    'classifier.3.weight',
    'classifier.4.weight',
    'classifier.4.bias',
    'classifier.6.weight'
]


for name, param in model.named_parameters():
    if name in params_to_update:
        print(name)
        param.requires_grad = True
    else:
        param.requires_grad = False


model = model.cuda()


def myCrossEntropyLoss(outputs, targets, weight):
    bceloss = nn.BCELoss(weight)
    outputs = (1-outputs)
    outputs = (outputs - outputs.min())/(outputs.max()-outputs.min())*(1-.9)+.9
    return bceloss(1 - outputs.prod(2).prod(2), targets)


ws = [1, 2, 2, 2, 2, 2]


optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), .001, [.9, .999])

import time

model.train()
for __ in tqdm(range(20)):
    start = time.time()
    train_dataloader.batch()
    end = time.time()
    print(end-start)
    optimizer_ft.zero_grad()
    outputs = model(batchX)
    loss = myCrossEntropyLoss(outputs, batchY, weight=torch.Tensor(ws).to(device))
    print(loss)
    loss.backward()
    optimizer_ft.step()
    del outputs, loss







