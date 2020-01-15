
import time
import pydicom
import numpy as np
from math import log, e
from queue import Queue
import threading
from random import random as rnd, shuffle
from scipy.stats import truncnorm, uniform

from torch import nn
from torchvision.transforms import Normalize
from torchvision.transforms import functional as F

from global_params import *


eps = (1.0 / 255.0)
ue = log((1.0 / eps) - 1.0)


class SigmoidWindowing(nn.Module):

    def __init__(self):
        super(SigmoidWindowing, self).__init__()

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


class DICOMPreprocessor:

    def __init__(self, augment=True):

        self.windower = SigmoidWindowing()

        self.augment = augment

        self.normalize = lambda x: F.normalize(x, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        if augment:
            random_resized_crop = lambda x, d: F.resized_crop(x, x.size[0] // 2 - d, x.size[1] // 2 - d, d * 2, d * 2,
                                                              size=(x.size[0], x.size[1]))

            brightness_params = truncnorm(1 - .08, 1 + .08), .5
            contrast_params = truncnorm(1 - .08, 1 + .08), .5
            random_resized_params = uniform(.7 * dimX // 2, .3 * dimY // 2), 0
            rotate_params = uniform(-30, 60), .3

            self.FT = [(F.adjust_brightness, brightness_params),
                       (F.adjust_contrast, contrast_params),
                       (random_resized_crop, random_resized_params),
                       (F.rotate, rotate_params)
                       ]

    def __call__(self, img_name):
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
        return self.normalize(F.to_tensor(x)).unsqueeze(0)


class Preprocessor(threading.Thread):

    def __init__(self, preprocessor=None, labels=True, idx=None, row=None):
        super(Preprocessor, self).__init__()
        self.preprocessor = preprocessor
        self.labels = labels
        self.idx = idx
        self.row = row
        self.display = False

    def run(self):

        global batchX, device, processed_in_batch

        img_name = self.row.filepath

        img = self.preprocessor.__call__(img_name)

        if self.labels:
            global batchY

            labels = torch.tensor(self.row.Label)
            batchX[self.idx], batchY[self.idx] = img.to(device), labels.to(device)
            del img, labels

        else:

            batchX[self.idx] = img.to(device)
            del img

        processed_in_batch += 1


class Controller:

    def __init__(self, data, labels=True, shuffle_=True):
        self.data = data
        self.labels = labels
        self.shuffle = shuffle_
        self.batch_size = batch_size
        self.n_batches = data.shape[0] // batch_size
        self.N = self.__len__()
        self.t_ct = 0  # total count

    def __len__(self):
        return len(self.data)

    def __shuffle__(self):
        self.data = self.data.reindex(np.random.permutation(self.data.index))
        self.data.index = np.arange(self.data.shape[0]).astype('int64')

    def __epoch__(self):
        self.t_ct = 0
        if self.shuffle:
            self.__shuffle__()

    def __itrn__(self):

        idx = int(self.t_ct % self.batch_size)
        row = self.data.loc[self.t_ct]
        self.t_ct += 1
        return idx, row


class DataLoader:

    def __init__(self, controller, preprocessor, num_workers=8):
        self.controller = controller
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.n_batches = self.controller.n_batches
        self.num_workers = num_workers
        self.workers = {i: Queue() for i in range(num_workers)}

    def epoch(self):
        self.controller.__epoch__()

    def shape(self):
        return self.controller.data.shape[0]

    def batch(self):
        global processed_in_batch
        processed_in_batch = 0
        for i in range(self.batch_size):
            worker = int(i % self.num_workers)
            row_idx, row_data = self.controller.__itrn__()
            processor = Preprocessor(self.preprocessor, idx=row_idx, row=row_data)
            self.workers[worker].put(processor.start())
        start = time.time()
        while processed_in_batch < batch_size or (end-start) < max_sleep:
            time.sleep(quicksleep)
            end = time.time()





