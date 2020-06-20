import os
import torch
import torch.nn as nn
from torch.utils.data import Dataloader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
from PIL import Image

def eval_model(model, dataloader, criterion, device, logger, tb_logger, show_figure):
  model.eval()
  running_loss = 0
  if show_figure:
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
      ax[idx, 0].imshow(label_img.reshape(256,512,3).astype(np.uint8), alpha=0.5])
      ax[idx, 1].imshow(img_tmp)
      ax[idx, 1].imshow(out_img.reshape(256,512,3).astype(np.uint8), alpha=0.5)
      ax[idx, 2].imshow(img_tmp)
  
  eval_loss = running_loss / idx
  if show_figure:
    return fig, eval_loss

  return None, eval_loss


def train_model(model, optimizer, criterion, training_dataloader, 
                validation_dataloader, device, epochs, batch_size, 
                lr, CHECKPOINT_PATH, model_name, logger, save_epoch=5,
                show_figure=True):

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
        #logger.add_scalar("Train Loss", running_loss / (idx + 1), global_step)

    epoch_loss = running_loss / (idx + 1)

    if (epoch + 1) % save_epoch == 0:
      fig, eval_loss = eval_model(model, validation_dataloader, criterion, logger, show_figure)
      logger.add_scalar("Eval Loss", eval_loss, global_step)
      logger.add_figure("Evaluation Results", fig, global_step)
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

  train_image_paths = []
  train_label_paths = []

  val_image_paths = []
  val_label_paths = []

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

  parser.add_argument('-e', '--epochs', metavar='E', type=int, default=10, 
                      help='Number of epochs', dest='epochs')
  parser.add_argument('-b', '--batch-size', metavar='B', type=int, default=1, 
                      help='Batch Size', dest='batchsize')
  parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, default=1e-4, 
                      help='Learning Rate', dest='lr')
  parser.add_argument('-m', '--model', metavar='M', type=str, default='FCN', 
                      help='Model for training', dest='model')
  parser.add_argument('-b', '--backbone', metavar='BB', type=str, default='resnet18', 
                      help='backbone for model', dest='backbone')
  parser.add_argument('-l', '--load', metavar='L', action='store_true')
  parser.add_argument('-c', '--checkpoint', metavar='C', type=str, 
                      help='checkpoint to load model', dest='checkpoint')
  #parser.add_argument('-op', '--optimizer', metavar='O', type=str, help='optimizer for training', dest='optimizer')
  #parser.add_argument('-cr', '--criterion', metavar='CR', type=str, help='criterion for training', dest='criterion')
  parser.add_argument('-d', '--dataset', metavar='D', type=str, 
                      help='root folder of dataset', dest='dataset')
  
  return parser.parse_args()

def main():
 
  args = get_args()
  logging.basicConfig(filename=args.model,
                      level=logging.INFO,
                      format="%(asctime)s:%(levelname)s:%(message)s")
  logger = logging.getLogger()

  device = torch.device('cuda', if torch.cuda.is_available() else 'cpu')
  logger.info(device)
  
  models = ['FCN', 'Unet', 'Deeplab']
  backbones = ['resnet18', 'resnet34', 'resnet50', 'mobilenet']
  assert args.model is in models, "Choose valid model"
  
  if args.model is 'FCN':
    assert args.backbone[:6] == 'resnet', 'Only resnet backbones supported for FCN'
    model = FCN(args.backbone).to(device) # TODO
  if args.model is 'Unet':
    model = Unet().to(device) # TODO
  if args.model is 'Deeplab':
    assert args.backbone[:9] == 'mobilenet', 'Only mobilenet backbones supported for Deeplab'
    model = Deeplab().to(device) # TODO
  
  logger.info("Model: {}, Backbone Used: {}".format(args.model, args.backbone))
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  criterion = torch.nn.CrossEntropyLoss()
  logger.info("Optimizer: Adam, Criterion/Loss Function: Cross Entropy Loss")
  
  if args.load:
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    criterion = checkpoint['criterion']
    logger.info("Model Loaded")
  
  ## Dataloaders 
  train_dataset, val_dataset = get_datasets(args.dataset)
  train_dataloader = Dataloader(train_dataset, batch_size=args.batchsize)
  val_dataloader = Dataloader(val_dataset, batch_size=args.batchsize, shuffle=True)
  logger.info("Dataset Used: {}".format(args.dataset))
  logger.info("Training Dataset Length: {}, Validation Dataset Length: {}".format( \ 
               len(train_dataloader), len(val_dataloader)))
  #log dataloader sizes
  train_model(model=model, optimizer=optimizer, criterion=criterion, 
              training_dataloader=train_dataloader, validation_dataloader=val_dataloader, 
              device=device, epochs=arg.epochs, lr=args.lr, CHECKPOINT_PATH=args.saveDir, 
              model_name=args.model, logger=logger, writer=writer, save_epoch=True, 
              show_figure=True)

if __name__ == 'main':
  main()
  
