import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import utils

class Cityscape(Dataset):
  '''
  Custom Dataset class for the Cityscape Dataset
  '''
  def __init__(self, image_paths, annotation_paths, transform=None, label_transform=None):
    self.images = []
    self.original_images = []
    self.annotations = []
    self.image_paths = image_paths
    self.annotation_paths = annotation_paths
    self.transform = transform
    self.label_transform = label_transform
    self.rgb = [0.485, 0.456, 0.406] 
    self.rgb_std= [0.229, 0.224, 0.225]
    self.rgb_cityscapes = [0.2838917598507514, 0.32513300997108185, 0.28689552631651594]
    self.rgb_std_cityscapes = [0.040622030307410795, 0.04200336505191565, 0.04646142476429996]
    print(" ===== Loading Dataloader ===== ")
    for image_path, label_path in tqdm(zip(self.image_paths, self.annotation_paths)):
      img = Image.open(image_path)
      mask = Image.open(label_path)#.convert('L')

      if self.transform:
        img = self.transform(img)
      else:
        ## print("====== performing default transforms for input images ======")
        img = TF.to_tensor(TF.resize(img, size=(256, 512), interpolation=Image.BILINEAR))
        img = TF.normalize(img, self.rgb, self.rgb_std)

      if self.label_transform:
        mask = self.label_transform(mask)
        mask = self.mask_to_label(np.array(mask))
      else:
        ## default is recommended 
        ## if required change the mask_to_label mapping
        ## print("===== performing default transforms for labels =====")
        mask = self.mask_to_label(np.array(TF.resize(mask, size=(256, 512), 
                                              interpolation=Image.NEAREST)))

      self.images.append(img)
      self.annotations.append(mask)

  def mask_to_label(self, mask):
    '''
    change default mask labels to your preferred mask labels
    Args:
      mask (np.ndarray): default labels
    Returns:
      torch.tensor: new labels as a tensor
    '''

    assert isinstance(mask, np.ndarray)
    new_mask = torch.zeros(mask.shape[0], mask.shape[1], dtype=torch.uint8)
    mapping_dict = utils.get_mapping_dict()
    for i in mapping_dict:
      new_mask[i==mask] = mapping_dict[i]

    return new_mask

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    img = self.images[idx]
    mask = self.annotations[idx]

    return img, mask
