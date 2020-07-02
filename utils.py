# convert label to color 
def get_label2color_dict():
  mapping_dict = {
    0: (128, 64, 128),
    1: (244, 35, 232),
    2: (70, 70, 70),
    3: (102, 102, 15),
    4: (190, 153, 15),
    5: (153, 153, 15),
    6: (250, 170, 30),
    7: (220, 220, 0),
    8: (107, 142, 35),
    9: (152, 251, 15),
    10: (0, 130, 180),
    11: (220, 20, 60),
    12: (0,0,0),  
    13: (0, 0, 0),
    14: (0, 0, 0),
    15: (0, 0, 0),  
    16: (0, 0, 0),
    17: (0, 0, 0),
    18: (0, 0, 0),
    19: (0, 0, 0),  
    20: (0, 0, 0),
    21: (0, 0, 0),
    22: (0, 0, 0),
    23: (0, 0, 0),
    24: (0, 0, 0),
    25: (0, 0, 0),
    26: (0, 0, 0),
    27: (0, 0, 0),
    28: (0, 0, 0),
    29: (0, 0, 0),
    30: (0, 0, 0),
    31: (0, 0, 0),
    32: (0, 0, 0),
    33: (0, 0, 0),
    -1: (0, 0, 0), 
    255: (0, 0, 0)
  }

  return mapping_dict

# convert Cityscape labels to 
# the ones we want to train
def get_mapping_dict():
  map_dict = {
    0: 0,  # unlabeled
    1: 0,  # ego vehicle
    2: 0,  # rect border
    3: 0,  # out of roi
    4: 0,  # static
    5: 0,  # dynamic
    6: 0,  # ground
    7: 1,  # road
    8: 2,  # sidewalk
    9: 0,  # parking
    10: 0,  # rail track
    11: 3,  # building
    12: 3,  # wall
    13: 3,  # fence
    14: 3,  # guard rail
    15: 3,  # bridge
    16: 0,  # tunnel
    17: 4,  # pole
    18: 4,  # polegroup
    19: 4,  # traffic light
    20: 4,  # traffic sign
    21: 5,  # vegetation
    22: 6,  # terrain
    23: 7,  # sky
    24: 8,  # person
    25: 8,  # rider
    26: 9,  # car
    27: 9,  # truck
    28: 9,  # bus
    29: 0,  # caravan
    30: 0,  # trailer
    31: 0,  # train
    32: 10,  # motorcycle
    33: 10,  # bicycle
    -1: -1  # licenseplate
  }
  return map_dict

def freeze_model(args, model):
    if args.model_name is 'FCN': # FCN16
      for idx, n, c in (model.named_children()):
          for params in c.parameters():
              params.requires_grad = False
        if idx == args.frozenlayers:
            break
      
      if !args.num_classes: 
        args.num_classes = 12

      model.Conv1x1 = nn.Conv2d(512, args.num_classes, kernel_size=(1,1), stride=(1,1))
      model.Conv1x1_x4 = nn.Conv2d(256, args.num_classes, kernel_size=(1,1), stride=(1,1))
      model.bn1 = nn.BatchNorm2d(args.num_classes)
      model.DCN2 = nn.ConvTranspose2d(args.num_classes, args.num_classes, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      model.dbn2 = nn.BatchNorm2d(args.num_classes)
      model.DCN16 = nn.ConvTranspose2d(args.num_classes, args.num_classes, kernel_size=(32, 32), stride=(16, 16), padding=(8, 8))
      model.dbn16 = nn.BatchNorm2d(args.num_classes)

    # TODO add mobilenet / unet transfer learning

    return model

