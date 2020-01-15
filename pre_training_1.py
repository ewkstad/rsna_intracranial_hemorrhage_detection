# Credit to https://www.kaggle.com/wfwiggins203 in https://www.kaggle.com/wfwiggins203/eda-dicom-tags-windowing-head-cts

import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
import re
import os
from pathlib import Path
import torch
from torch import nn
from torch.nn import Sigmoid
from torchvision.transforms import functional as F
from torchvision.transforms import Normalize
from math import log
from numpy import e
plt.style.use('grayscale')


device = torch.cuda.current_device()
device

ROOT_DIR = "C:/Users/evbruh/Downloads/rsna-intracranial-hemorrhage-detection/"
TRAIN_DIR = ROOT_DIR + 'stage_1_train_images'
TEST_DIR = ROOT_DIR + 'stage_1_test_images'

all_df = pd.read_csv(ROOT_DIR + 'stage_1_train.csv')

all_df[['ID', 'Subtype']] = all_df['ID'].str.rsplit(pat='_', n=1, expand=True)


def fix_id(img_id, img_dir=TRAIN_DIR):
    if not re.match(r'ID_[a-z0-9]+', img_id):
        sop = re.search(r'[a-z0-9]+', img_id)
        if sop:
            img_id_new = f'ID_{sop[0]}'
            return img_id_new
        else:
            print(img_id)
    return img_id


def id_to_filepath(img_id, img_dir=TRAIN_DIR):
    filepath = f'{img_dir}/{img_id}.dcm'  # pydicom doesn't play nice with Path objects
    if os.path.exists(filepath):
        return filepath
    else:
        return 'DNE'


def get_patient_data(filepath):
    if filepath != 'DNE':
        dcm_data = pydicom.dcmread(filepath, stop_before_pixels=True)
        return dcm_data.PatientID, dcm_data.StudyInstanceUID, dcm_data.SeriesInstanceUID

all_df['ID'] = all_df['ID'].apply(fix_id)

all_df = all_df.pivot_table(index='ID', columns='Subtype').reset_index()

all_df['filepath'] = all_df['ID'].apply(id_to_filepath)

train_df = all_df[:int(.9*all_df.shape[0])]
val_df = all_df[int(.9*all_df.shape[0]):]

hem_types = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

import random
from random import random as rnd, shuffle
from torchvision import transforms
from scipy.stats import truncnorm, uniform
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, resized_crop, rotate

eps = (1.0 / 255.0)
ue = log((1.0 / eps) - 1.0)


class sigmoid_windowing(nn.Module):

    def __init__(self):
        super(sigmoid_windowing, self).__init__()

        self.eps = eps
        self.ue = ue
        self.e = e

        self.w0 = (2 / 80) * ue
        self.b0 = ((-2 * 40) / 80) * ue

        self.w1 = (2 / 200) * ue
        self.b1 = ((-2 * 80) / 200) * ue

        self.w2 = (2 / 2000) * ue
        self.b2 = ((-2 * 600) / 2000) * ue

        self.vdim = (512, 512, -1)

        self.n = Normalize(mean=[0, 0, 0],
                           std=[1, 1, 1])

    def forward(self, dcm):
        y = torch.Tensor(dcm.pixel_array.astype('float32')).to(device) * dcm.RescaleSlope + dcm.RescaleIntercept

        y0 = (1.0 / (1 + self.e ** (-1.0 * (self.w0 * y + self.b0)))).view(self.vdim)
        y1 = (1.0 / (1 + self.e ** (-1.0 * (self.w1 * y + self.b1)))).view(self.vdim)
        y2 = (1.0 / (1 + self.e ** (-1.0 * (self.w2 * y + self.b2)))).view(self.vdim)

        return self.n(torch.cat((y0, y1, y2), 2))


class DICOMPreprocessor():

    def __init__(self, augment=True):

        self.windower = sigmoid_windowing()

        self.augment = augment

        self.normalize = lambda x: F.normalize(x, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        if augment:
            random_resized_crop = lambda x, d: F.resized_crop(x, x.size[0] // 2 - d, x.size[1] // 2 - d, d * 2, d * 2,
                                                              size=(x.size[0], x.size[1]))

            brightness_params = truncnorm(1 - .08, 1 + .08), .5
            contrast_params = truncnorm(1 - .08, 1 + .08), .5
            random_resized_params = uniform(.7 * 512 // 2, .3 * 512 // 2), 0
            rotate_params = uniform(-30, 60), .3

            self.FT = [(adjust_brightness, brightness_params),
                       (adjust_contrast, contrast_params),
                       (random_resized_crop, random_resized_params),
                       (rotate, rotate_params)
                       ]

    def __call__(self, img_name, output_tensor=True):
        dcm = pydicom.read_file(img_name[0])
        x = self.windower(dcm)
        if self.augment:
            x = F.to_pil_image((255 * x).cpu().numpy().astype('uint8'))
            if rnd() < .5:
                x = F.hflip(x)
            if rnd() < .5:
                x = F.vflip(x)
            shuffle(self.FT)
            for ft, (d, p) in self.FT:
                if rnd() < p:
                    x = ft(x, d.rvs())
        if output_tensor:
            return self.normalize(F.to_tensor(x)).unsqueeze(0)
        return x


train_dicom_preprocessor = DICOMPreprocessor(augment=True)
val_dicom_preprocessor = DICOMPreprocessor(augment=True)
test_dicom_preprocessor = DICOMPreprocessor(augment=False)

import torch.nn as nn
from torchvision import models
import torch.nn.functional as nnF


#
def get_dcnn_base(arch, pretrained):
    if arch == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        model = nn.Sequential(*list(model.features)[:-1])
    else:
        print('No {}!'.format(arch))
        raise ValueError
    return model


#
class DenseNet121(nn.Module):

    def __init__(self, num_classes=6, out_hw=16, pretrained=True):
        super(DenseNet121, self).__init__()
        self.features = get_dcnn_base('densenet121', pretrained)
        self.up = nn.Upsample(size=(out_hw, out_hw), mode='bilinear')
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(6)

    def forward(self, x) -> 'PxPxK tensor prediction':
        x = self.features(x)
        x = self.up(x)
        x = self.bn1(x)
        x = nnF.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = nnF.relu(x)
        x = self.conv2(x)
        x = self.bn3(x)
        return x

model = DenseNet121()
model = model.cuda()

t_layer = 10
for i, child in enumerate(model.features.children()):
    if i < t_layer:
        for param in child.parameters():
            param.requires_grad = False

t_layer = 0
for i, child in enumerate(model.children()):
    if i > t_layer:
        for param in child.parameters():
            param.requires_grad = True

from torch import optim
optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

def myCrossEntropyLoss(outputs, targets, weight):
    bceloss = nn.BCELoss(weight)
    return bceloss(1 - (1-outputs).reshape(-1,6,256).prod(2), targets)


import random
import threading
from queue import Queue
from random import random as rnd, shuffle
from torchvision import transforms
from scipy.stats import truncnorm, uniform
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, resized_crop, rotate

eps = (1.0 / 255.0)
ue = log((1.0 / eps) - 1.0)

batch_size = 32
batch_dim = (batch_size, 3, 512, 512)
batchX = torch.zeros(batch_dim).to(device)
batchY = torch.zeros((batch_size, 6)).to(device)


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self):
        super(StoppableThread, self).__init__()
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class Preprocessor(threading.Thread):

    def __init__(self, preprocessor=None, labels=True, idx=None, row=None):
        super(Preprocessor, self).__init__()
        self.preprocessor = preprocessor
        self.labels = labels
        self.idx = idx
        self.row = row
        self.display = False
        self.device = torch.cuda.current_device()

    def run(self):

        global batchX

        img_name = self.row.filepath

        img = self.preprocessor.__call__(img_name, output_tensor=not self.display)

        if self.labels:
            global batchY

            labels = torch.tensor(self.row.Label)
            batchX[self.idx], batchY[self.idx] = img.to(self.device), labels.to(self.device)
            del img, labels

        else:

            batchX[self.idx] = img.to(self.device)
            del img


class Controller():

    def __init__(self, data, labels=True, batch_size=32):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.N = self.__len__()
        self.t_ct = 0  # total count

    def __len__(self):
        return len(self.data)

    def __shuffle__(self):  # TODO
        pass

    def __epoch__(self):
        self.t_ct = 0

    def __itrn__(self):
        idx = int(self.t_ct % self.batch_size)

        row = self.data.loc[self.t_ct]

        self.t_ct += 1

        return idx, row


class Worker(threading.Thread):

    def __init__(self, q):
        self.q = q
        super(Worker, self).__init__()

    def run(self, process):
        self.q.put(process)


class DataLoader:

    def __init__(self, controller, preprocessor, batch_size=32, shuffle=True, num_workers=8):
        self.controller = controller
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.workers = {i: Worker(Queue()) for i in range(num_workers)}

    def batch(self):
        for i in range(self.batch_size):
            worker = int(i % self.num_workers)
            row_idx, row_data = self.controller.__itrn__()
            processor = Preprocessor(self.preprocessor, idx=row_idx, row=row_data)
            self.workers[worker].q.put(processor.start())

train_controller = Controller(data=train_df)
val_controller = Controller(data=val_df)

train_dataloader = DataLoader(train_controller, train_dicom_preprocessor)
val_dataloader = DataLoader(val_controller, val_dicom_preprocessor)

from tqdm import tqdm
from torch.autograd import Variable

model.train()
for __ in tqdm(range(100)):
    train_dataloader.batch()
    time.sleep(2)
    optimizer_ft.zero_grad()
    outputs = model(Variable(batchX))
    loss = myCrossEntropyLoss(outputs, Variable(batchY), weight=torch.Tensor([2, 1, 1, 1, 1, 1]).to(device))
    print(loss)
    loss.backward()
    optimizer_ft.step()
    del outputs, loss







