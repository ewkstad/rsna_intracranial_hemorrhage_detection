'''
author: Gu Wang
source: https://github.com/wangg12/intel-cervical-cancer/blob/master/train_eval_funcs.py
'''
from __future__ import division, absolute_import, print_function

import os
import copy
import time
from tqdm import tqdm
from numpy import inf
import pickle
import torch.nn.functional as F

import torch

from utils import KaggleLogLoss, optim_scheduler_ft, log_value

from global_params import batchX, batchY, ROOT_DIR, LOGGING_PATH, exp


def train_model(args,
                model,
                criterion,
                dset_loaders,
                dset_sizes,
                rlrp_agent,
                num_epochs,
                epoch_trained=0,
                best_epoch_=-1,
                best_model_logloss_=None,
                best_model_acc_=0.0,
                best_model_=None):
  '''
  optim_scheduler: a function which returns an optimizer object when called as optim_scheduler(model, epoch)
      This is useful when we want to change the learning rate or restrict the parameters we want to optimize.
  '''

  # ROOKIE SHIT CODE HERE
  log_dict = None
  if os.path.isfile(LOGGING_PATH):
    log_dict = pickle.load(open(LOGGING_PATH, 'rb'))
    if type(log_dict) is not dict:
      log_dict = None
  if log_dict is None:
    log_dict = dict()

  since = time.time()

  best_model = best_model_
  best_model_acc = best_model_acc_
  best_model_logloss = best_model_logloss_
  best_epoch = best_epoch_

  init_lr = args['lr']
  lr = args['lr']
  epoch_loss = inf

  LogLoss = KaggleLogLoss()


  for epoch in range(epoch_trained+1, epoch_trained+1+num_epochs):
    print('Epoch {}/{}'.format(epoch, epoch_trained+num_epochs))
    print('-' * 10)

    val_acc = 0.0
    val_logloss = None
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      dset_loaders[phase].epoch()
      n_batches = dset_loaders[phase].n_batches
      if phase == 'train':
        model.train()
        # TODO: Have to pass loss as an argument. Initialize, then have to grab a copy in the loop.
        optimizer, lr, init_lr, rlrp_agent = optim_scheduler_ft(model,
                                                                epoch,
                                                                metrics=epoch_loss,
                                                                optimizer_name=args['optimizer'],
                                                                init_lr=init_lr,
                                                                old_lr=lr,
                                                                lr_decay_factor=args['lr_decay_factor'],
                                                                lr_decay_epoch=args['lr_decay_epoch'],
                                                                momentum=args['momentum'],
                                                                weight_decay=args['weight_decay'],
                                                                warmup=args['warmup'],
                                                                warm_lr=args['warm_lr'],
                                                                warm_epochs=args['warm_epochs'],
                                                                warmup_type=args['warmup_type'],
                                                                reduce_on_plateau_agent=rlrp_agent)

        log_value(log_dict, 'lr', lr, step=epoch)
      elif phase == 'val':
        model.eval()

      running_loss = 0.0
      running_log_loss = 0.0
      running_corrects = 0

      # Iterate over data.
      step = 1
      for __ in tqdm(range(n_batches)):
        dset_loaders[phase].batch()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward =====================================================================
        if phase == 'train':
          outputs = model(batchX)

        elif phase == 'val':
          with torch.no_grad():
            outputs = model(batchX)

        _, preds = torch.max(outputs, dim=1)
        loss = criterion(outputs, batchY)
        # TODO: Include post-processing function for localization framework.
        loss_log = LogLoss(outputs, batchY)

        # backward + optimize only if in training phase =====================================
        if phase == 'train':
          loss.backward()
          optimizer.step()

        # statistics
        running_loss += loss.item()*batchX.shape[0]
        running_log_loss += loss_log.item()*batchX.shape[0]
        running_corrects += torch.sum(preds == torch.max(batchY, dim=1)[1])

        global_step = (epoch-1)*n_batches + step

        mem_cached = torch.cuda.max_memory_cached()


        log_value(log_dict, phase+'_{}'.format(args['loss']), loss.item(), step = global_step)
        log_value(log_dict, phase+'_log_loss', loss_log.item(), step = global_step)
        log_value(log_dict,
                  phase+'_acc',
                  torch.mean((preds == torch.max(batchY, dim=1)[1]).type_as(torch.FloatTensor())),
                  step=global_step)
        log_value(log_dict, 'cuda_memory_cached', mem_cached, step = global_step)
        print('{} {}_Loss: {:.4f}, Log_loss: {:.4f}, Memory_cached: {}'.format(phase, exp, loss.item(), loss_log.item(), mem_cached))
        step += 1

        # free the graph to avoid memory increase
        del outputs, loss, loss_log

      epoch_loss = running_loss / dset_sizes[phase]
      epoch_log_loss = running_log_loss / dset_sizes[phase]
      epoch_acc = running_corrects / dset_sizes[phase]

      print('{} {}_Loss: {:.4f}, Log_loss: {:.4f}, Acc: {:.4f}'.format(
                phase, exp, epoch_loss, epoch_log_loss, epoch_acc))
      log_value(log_dict, 'epoch{}_{}'.format(phase, args['loss']), epoch_loss, step=epoch)
      log_value(log_dict, 'epoch{}_log_loss'.format(phase), epoch_log_loss, step=epoch)
      log_value(log_dict, 'epoch{}_acc'.format(phase), epoch_acc, step=epoch)

      if phase == 'val':
        val_acc = epoch_acc
        val_logloss = epoch_log_loss
      # deep copy the model
      print(epoch_log_loss, best_model_logloss)
      print(type(epoch_log_loss), type(best_model_logloss))
      if phase == 'val' and (best_model_logloss is None or epoch_log_loss < best_model_logloss):
        best_model_acc = epoch_acc
        best_model_logloss = epoch_log_loss
        best_epoch = epoch
        best_model = copy.deepcopy(model)

    # Saving log dictionary every epoch
    pickle.dump(log_dict, open(LOGGING_PATH, 'wb'))

    # do checkpointing
    if epoch % args['ckpt_epoch'] == 0:
      save_checkpoint(state={'epoch': epoch,
                             'val_logloss':val_logloss,
                             'val_acc':val_acc,
                             'state_dict':model.state_dict()},
              save_path='{0}/{1}_epoch_{2}.pth'.format(ROOT_DIR, exp, epoch))
      save_checkpoint(state={'epoch': best_epoch,
                             'val_logloss':best_model_logloss,
                             'val_acc':best_model_acc,
                             'state_dict':best_model.state_dict()},
              save_path='{0}/{1}_best_model.pth'.format(ROOT_DIR, exp))
    print()

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
                  time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}, best epoch: {}'.format(best_model_acc, best_epoch))
  return best_model


def evaluate_model(args, model, criterion, log_loss, dset_loaders, dset_sizes):
  '''
  evaluate per-class accuracy
  '''
  running_loss = 0.0
  running_log_loss = 0.0
  total_corrects = 0

  total_type1 = 0
  corrects_type1 = 0
  total_type2 = 0
  corrects_type2 = 0
  total_type3 = 0
  corrects_type3 = 0

  # switch to evaluate mode
  model.eval()

  n_batches = dset_loaders['val'].controller.n_batches

  for __ in tqdm(n_batches):

    dset_loaders['val'].batch()

    # forward
    with torch.no_grad():
      outputs = model(batchX)

    _, preds = torch.max(outputs, dim=1)
    if args['CE_Loss']:
      loss = criterion(outputs, batchY)
    else:
      loss = criterion(outputs, batchY)

    loss_log = log_loss(outputs, batchY)

    # statistics
    running_loss += loss.item()*batchX.shape[0]
    running_log_loss += loss_log.item()*batchX.shape[0]
    total_corrects += torch.sum(preds == batchY)

  evaluate_loss = running_loss / dset_sizes['val']
  evaluate_log_loss = running_log_loss / dset_sizes['val']
  evaluate_total_acc = total_corrects / dset_sizes['val']

  print('Evaluation results')
  print('-' * 10)
  if args['CE_Loss']:
    print('CE_Loss: {:.4f}, Log_loss: {:.4f}, Acc: {:.4f}'.format(
                  evaluate_loss, evaluate_log_loss, evaluate_total_acc))
  else:
    print('BCE_Loss: {:.4f}, Log_loss: {:.4f}, Acc: {:.4f}'.format(
                  evaluate_loss, evaluate_log_loss, evaluate_total_acc))


def save_checkpoint(state, save_path='checkpoint.pth'):
  '''
  state: dict, {'epoch': epoch,
               'val_logloss':val_logloss,
               'val_acc':val_acc,
               'state_dict':model.state_dict()}
  '''
  save_dir = os.path.split(save_path)[0]
  if os.path.isdir(save_dir) and (not os.path.exists(save_dir)):
    os.makedirs(save_dir)
  torch.save(state, save_path)
  # if is_best:
  #   shutil.copyfile(save_path, os.path.join(save_dir ,'model_best.pth'))


def resume_checkpoint(model, ckpt_path='checkpoint.pth'):
  '''
  state: dict, {'epoch': epoch,
               'val_logloss':val_logloss,
               'val_acc':val_acc,
               'state_dict':model.state_dict()}
  '''
  state = torch.load(ckpt_path)
  best_model = copy.deepcopy(model)
  model.load_state_dict(state['state_dict'])
  epoch = state['epoch']

  # load the best modle
  best_path = os.path.join(os.path.dirname(ckpt_path),'best_model.pth')
  if os.path.isfile(best_path):
    state = torch.load(best_path)
    best_model.load_state_dict(state['state_dict'])
    best_epoch = state['epoch']
    best_model_acc = state['val_acc']
    best_model_logloss = state['val_logloss']
  else:
    print('checkpoint file of best model does not exist!')

  return epoch, model, best_epoch, best_model_logloss, best_model_acc, best_model