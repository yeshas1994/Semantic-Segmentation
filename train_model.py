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
from eval_model import eval_model

def train_model(model, optimizer, criterion, scheduler, training_dataloader, 
                validation_dataloader, device, epochs, batch_size, 
                lr, CHECKPOINT_PATH, model_name, logger, writer, 
                save_epoch=5, show_figure=True, save_model=True):

  logger.info("===== START TRAINING =====")
  model.train()
  global_step = 0
  for epoch in range(epochs):
    running_loss = 0
    epoch_loss = 0
    for idx, (img, label) in enumerate(training_dataloader):
      img = img.to(device)
      label = label.to(device)

      out = model(img)
      loss = criterion(out, label.long())
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      global_step += 1

      if (global_step) % 500 == 0:
        log = '=== Step:{}, Train_Loss: {:.5f} ==='.format(global_step, running_loss / (idx + 1))
        logger.info(log)
        writer.add_scalar("Train_Loss", running_loss / (idx + 1), global_step)

    epoch_loss = running_loss / (idx + 1)
    scheduler.step(epoch_loss)
    print(optimizer.param_groups[0]['lr']) 
    if (epoch + 1) % save_epoch == 0:
      fig, eval_loss = eval_model(model, validation_dataloader, criterion, device, logger, writer, show_figure)
      writer.add_scalar("Eval_Loss", eval_loss, global_step)
      writer.add_figure("Evaluation_Results", fig, global_step)
      log = 'Epoch: {}, Train_Loss: {:.5f}, Eval_Loss: {:.5f}'.format(epoch+1, running_loss/(idx+1), eval_loss)
      logger.info(log)
      if save_model:
        torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'criterion': criterion,
          'global_step': global_step
          }, CHECKPOINT_PATH + model_name + '_{}.pth'.format(epoch+1))
    else:
      log = 'Epoch: {}, Train_Loss: {:.5f}'.format(epoch+1, running_loss/(idx+1))
      logger.info(log)


def get_datasets(dataset_root, train=True, val=True, test=False):
  images_root = os.path.join(dataset_root, 'images')
  annotations_root = os.path.join(dataset_root, 'annotations')

  train_dataset = None
  val_dataset = None

  train_images_paths = []
  train_labels_paths = []

  val_images_paths = []
  val_labels_paths = []

  if train:
    train_images_root = os.path.join(images_root, 'training')
    train_labels_root = os.path.join(annotations_root, 'training')
    for image, label in zip(os.listdir(train_images_root), os.listdir(train_labels_root)):
      train_images_paths.append(os.path.join(train_images_root, image))
      train_labels_paths.append(os.path.join(train_labels_root, label))
  if val:
    val_images_root = os.path.join(images_root, 'validation')
    val_labels_root = os.path.join(annotations_root, 'validation')
    for image, label in zip(os.listdir(val_images_root), os.listdir(val_labels_root)):
      val_images_paths.append(os.path.join(val_images_root, image))
      val_labels_paths.append(os.path.join(val_labels_root, label))
 
  if train:
    train_dataset = Cityscape(train_images_paths, train_labels_paths)
  if val:
    val_dataset = Cityscape(val_images_paths, val_labels_paths)

  return train_dataset, val_dataset

def get_args():
  parser = argparse.ArgumentParser(description='Deep Learning & Transfer Learning for \
                                      Cityscapes with a variety of architectures')

  parser.add_argument('-e', '--epochs', metavar='E', type=int, default=50, 
                      help='Number of epochs', dest='epochs')
  parser.add_argument('-b', '--batch-size', metavar='B', type=int, default=1, 
                      help='Batch Size', dest='batchsize')
  parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, default=1e-4, 
                      help='Learning Rate', dest='lr')
  parser.add_argument('-m', '--model', metavar='M', type=str, default='FCN', 
                      help='Model for training', dest='model')
  parser.add_argument('-bk', '--backbone', metavar='BB', type=str, default='resnet18', 
                      help='backbone for model', dest='backbone')
  parser.add_argument('-l', '--load', action='store_true')
  parser.add_argument('-c', '--checkpoint', metavar='C', type=str, 
                      help='checkpoint to load model', dest='checkpoint')
  #parser.add_argument('-op', '--optimizer', metavar='O', type=str, help='optimizer for training', dest='optimizer')
  #parser.add_argument('-cr', '--criterion', metavar='CR', type=str, help='criterion for training', dest='criterion')
  parser.add_argument('-d', '--dataset', metavar='D', type=str, default='data/Cityscape', 
                      help='root folder of dataset', dest='dataset')
  parser.add_argument('-s', '--saveDir', metavar='S', type=str, default='saved_models/',
                      help='folder to save drive', dest='saveDir')
  
  return parser.parse_args()

def main():
 
  print('start')
  args = get_args()
  logging.basicConfig(filename=args.model + '.log',
                      level=logging.INFO,
                      format="%(asctime)s: %(message)s")
  logger = logging.getLogger()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logger.info(device)
  
  models = ['FCN', 'Unet', 'Deeplab']
  backbones = ['resnet18', 'resnet34', 'resnet50', 'mobilenet']
  assert args.model in models, "Choose valid model"
  
  if args.model is 'FCN':
    assert args.backbone[:6] == 'resnet', 'Only resnet backbones supported for FCN'
    model = FCN.FCN16(args.backbone, num_classes=11).to(device) 
  if args.model is 'Unet':
    model = Unet().to(device) 
  if args.model is 'Deeplab':
    assert args.backbone[:9] == 'mobilenet', 'Only mobilenet backbones supported for Deeplab'
    model = Deeplab().to(device) # TODO
  
  logger.info("Model: {}, Backbone Used: {}".format(args.model, args.backbone))
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  criterion = torch.nn.CrossEntropyLoss()
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=1e-3) 
  logger.info("Optimizer: Adam")
  logger.info("Criterion/Loss Function: Cross Entropy Loss")
  logger.info("Scheduler: ReduceLROnPlateau")
  logger.info("Learning Rate: {}".format(args.lr))
  
  if args.load:
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    criterion = checkpoint['criterion']
    logger.info("Model Loaded")
  
  ## Dataloaders 
  train_dataset, val_dataset = get_datasets(args.dataset, train=True)
  logger.info("Dataset Used: {}".format(args.dataset))
  if train_dataset:
    train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize)
    logger.info("Training Dataset Length: {}".format(len(train_dataloader)))
  if val_dataset:
    val_dataloader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=True)
    logger.info("Validation Dataset Length: {}".format(len(val_dataloader)))
  
  #if not os.path.exists(args.saveDir):
  writer = SummaryWriter(os.path.join(args.saveDir, args.model+'_log'))
  train_model(model=model, 
              optimizer=optimizer, 
              criterion=criterion,
              scheduler=scheduler,
              training_dataloader=train_dataloader, 
              validation_dataloader=val_dataloader, 
              device=device, 
              epochs=args.epochs, 
              batch_size=args.batchsize,
              lr=args.lr, 
              CHECKPOINT_PATH=args.saveDir, 
              model_name=args.model, 
              logger=logger, 
              writer=writer, 
              save_epoch=5, 
              show_figure=True,
              save_model=True)

if __name__ == '__main__':
  main()
  
