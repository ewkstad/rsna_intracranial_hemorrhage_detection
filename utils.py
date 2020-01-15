'''
author: Gu Wang
source: https://github.com/wangg12/intel-cervical-cancer/blob/master/utils.py
'''

import torch
from torch import nn
from torch import optim
import torch.nn.init as init
from copy import deepcopy
from lr_scheduler import warmup_scheduler, cosine_anneal_schedule


class KaggleLogLoss(nn.Module):
  """kaggle log loss function.
  -1/N sum(y*log(y_pred))
  the submitted probabilities are divided by row_sum,
      and then max(min(p, 1-1e-15), 1e-15), can be achieved by torch.clamp
  TODO: not sure the order of these two operations
  """
  def __init__(self):
    super(KaggleLogLoss, self).__init__()

  def forward(self, y_pred, y_true_one_hot):
    '''
    y_pred: [B,C],
    y_true_one_hot: [B,C], Variable torch.LongTensor
    '''
    ## y_pred has passed through softmax
    # do not average over batch size here
    loss = - torch.mean(torch.sum(y_true_one_hot * torch.log(torch.clamp(y_pred, min=1e-15, max=1-1e-15)), 1))

    # loss = - torch.mean(y_true_one_hot * F.log_softmax(y_pred)) # y_pred should not be passed through softmax
    # loss = - torch.mean(y_true_one_hot * torch.log(y_pred + 1e-15)) # y_pred has passed through softmax
    # print(loss.data[0])
    return loss


def weights_init(m):
  # classname = m.__class__.__name__
  if isinstance(m, nn.Conv2d):
    #print('init conv2d')
    #init.xavier_uniform(m.weight.data, gain=np.sqrt(2.0))
    init.kaiming_uniform(m.weight.data, mode='fan_in')
    # m.weight.data.normal_(0.0, 0.02)
  if isinstance(m, nn.Linear):
    #print('init fc')
    init.kaiming_uniform(m.weight.data, mode='fan_in')
    # size = m.weight.size()
    # fan_out = size[0] # number of rows
    # fan_in = size[1] # number of columns
    # variance = np.sqrt(2.0/(fan_in + fan_out))
    # m.weight.data.uniform_(0.0, variance)

# Assumes the passed model's trainable parameters have already been set.
def optim_scheduler_ft(model, epoch, metrics=None, optimizer_name='rmsprop',
                       init_lr=0.001, old_lr=None, lr_decay_epoch=10, lr_decay_factor=0.9,
                       momentum=0.9, weight_decay=1e-4,
                       beta1=0.9,
                       warmup=False, warm_lr=1e-4, warm_epochs=5, warmup_type='constant',
                       cos_schedule=False, cos_schedule_params=None, reduce_on_plateau_agent=None):
  '''exponentially decrease the learning rate once every few epochs
  beta1: beta1 for adam, default is 0.9
  cos_schedule_params: for example: {'T': 100, 'M': 10, 'init_lr': 0.1}
  ----
  optimizer: the re-scheduled optimizer
  '''
  # TODO: Reset plateau reducer when some other scheduler reduces it.
  if epoch == 0:
    old_lr = deepcopy(init_lr)

  if cos_schedule == False:
    if warmup:
      if epoch <= warm_epochs:
        new_lr = warmup_scheduler(epoch, warm_lr=warm_lr, warm_epochs=warm_epochs, warmup_type=warmup_type, target_lr=old_lr)
      else:
        new_lr = init_lr * (lr_decay_factor**((epoch - 1 - warm_epochs) // lr_decay_epoch))
        if (epoch - 1 - warm_epochs) % lr_decay_epoch == 0:
          print('learning rate is set to {}'.format(new_lr))
    else:
      new_deterministic_lr = init_lr * (lr_decay_factor**((epoch-1) // lr_decay_epoch))
      if reduce_on_plateau_agent is not None:
        new_adaptive_lr = reduce_on_plateau_agent.step(metrics, epoch, old_lr)
        if new_deterministic_lr < old_lr:
          reduce_on_plateau_agent.reset()
          new_lr = new_deterministic_lr
        elif new_adaptive_lr < old_lr:
          init_lr = new_adaptive_lr
          new_lr = new_adaptive_lr
      else:
        new_lr = new_deterministic_lr

      if (epoch - 1) % lr_decay_epoch == 0:
        print('learning rate is set to {}'.format(new_lr))

  else: # cosine schedule
    if warmup:
      if epoch<=warm_epochs:
        new_lr = warmup_scheduler(epoch, warm_lr=warm_lr, warm_epochs=warm_epochs, warmup_type=warmup_type, target_lr=init_lr)
      else:
        new_lr = cosine_anneal_schedule(epoch-1-warm_epochs, **cos_schedule_params)
    else:
      new_lr = cosine_anneal_schedule(epoch-1, **cos_schedule_params)
  print('epoch:{}, lr:{}'.format(epoch, new_lr))

  optimizer_name = optimizer_name.lower()
  if optimizer_name == 'adam':
      optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=new_lr, betas=(beta1, 0.999), eps=1e-08, weight_decay=0)
  elif optimizer_name == 'adadelta':
      optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=new_lr, rho=0.9, eps=1e-06, weight_decay=0)
  elif optimizer_name == 'rmsprop':
      optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=new_lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
  elif optimizer_name == 'nag':
      optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=new_lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
  elif optimizer_name == 'sgd':
      optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=new_lr, momentum=momentum, weight_decay=weight_decay)
  else:
      optimizer = None
      print("No optimizer: {}!".format(optimizer_name))
  return optimizer, new_lr, init_lr, reduce_on_plateau_agent


# TODO: LOT OF PROBLEMS HERE, CHEAPEST WAY OF GETTING AROUND NOT USING TENSORBOARD.
def log_value(dc, key, value, step):
  if key not in dc:
    dc[key] = dict()
  try:
    dc[key][step] = float(value)

  except Exception as e:
    print('e is ', e)

  return dc

