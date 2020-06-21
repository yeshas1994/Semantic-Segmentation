import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from models import FCN
import numpy as np
import logging
import argparse
from PIL import Image
from datasets.dataset import Cityscape
import matplotlib.pyplot as plt
import utils

def eval_model(model, dataloader, criterion, device, logger, tb_logger, show_figure):
  model.eval()
  running_loss = 0
  if show_figure:
    label2color = utils.get_label2color_dict()
    fig, ax = plt.subplots(20,3, figsize=(20,60))
    ax[0,0].set_title("Ground Truth")
    ax[0,1].set_title("Prediction (Overlay)")
    ax[0,2].set_title("Original Image")

  with torch.no_grad():
    for idx, (img, label) in enumerate(dataloader):
      x = img.numpy()
      img, label = img.to(device), label.to(device)
      out = model(img)
      loss = criterion(out, label.long())
      pred = torch.argmax(out, dim=1).detach().cpu().numpy()
      label = label.detach().cpu().numpy()
      running_loss += loss.item()
      
      if show_figure:
        out_img = np.zeros((256*512, 3))
        label_img = np.zeros((256*512, 3))
        for j, (id_pred, id_label) in enumerate(zip(pred.flatten(), label.flatten())):
          out_img[j] = np.array(label2color[id_pred])
          label_img[j] = np.array(label2color[id_label])
      if idx == 10:
        break
    if show_figure:
      # img_tmp = utils.unnormalize(x)
      img_tmp = np.transpose(x[0], axes=[1, 2, 0])
      img_tmp *= (0.229, 0.224, 0.225)
      img_tmp += (0.485, 0.456, 0.406)
      img_tmp *= 255.0
      img_tmp = img_tmp.astype(np.uint8)

      ax[idx, 0].imshow(img_tmp)
      ax[idx, 0].imshow(label_img.reshape(256,512,3).astype(np.uint8), alpha=0.5)
      ax[idx, 1].imshow(img_tmp)
      ax[idx, 1].imshow(out_img.reshape(256,512,3).astype(np.uint8), alpha=0.5)
      ax[idx, 2].imshow(img_tmp)
  
  eval_loss = running_loss / idx
  if show_figure:
    return fig, eval_loss

  return None, eval_loss

  
