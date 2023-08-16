import numpy as np
import torch
import os

def get_mask_name(info_dict):
  mask_type = info_dict["mask_type"]
  
  if mask_type == 'center':
    return mask_type
  elif mask_type == "checkerboard" or mask_type == "checkerboard_alt":
    return "{mask_type}_blocks{num_blocks}".format(
      mask_type=mask_type, num_blocks=info_dict["checkerboard_num_blocks"]
    )
  elif mask_type == "random":
    return "{mask_type}_patch{patch_size}_ratio{mask_ratio}".format(
      mask_type=mask_type, 
      patch_size=info_dict["rand_patch_size"],
      mask_ratio=info_dict["rand_mask_ratio"],
    )
  else:
    raise NotImplementedError

def save_mask(mask, info_dict, save_dir, compress=True, identifier=0):
  os.makedirs(save_dir, exist_ok=True)

  mask = mask.cpu().detach()
  if compress:
    mask = mask[0,0,:,:]

  save_path = "{save_dir}/{mask_name}_res{image_size}_{identifier}.pkl".format(
    save_dir=save_dir, mask_name=get_mask_name(info_dict), 
    image_size=info_dict["image_size"], identifier=identifier
  ) 
  torch.save(mask, save_path)
  return

def get_mask(mask_type, 
             image_size, 
             batch_size, 
             num_channels=3,
             mask_file_path=None,
             checkerboard_num_blocks=8, 
             rand_patch_size=4, 
             rand_mask_ratio=0.5, 
             maskgen=None, 
             maskgen_offset=0):

  if mask_type == "random":
    if maskgen is None:
      maskgen = MaskGenerator(input_size=image_size, 
                              mask_patch_size=rand_patch_size,
                              model_patch_size=1, 
                              mask_ratio=rand_mask_ratio)
    return torch.tensor(maskgen()).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_channels, 1, 1)
    
  elif mask_type == "center":
    mask = torch.ones((batch_size, num_channels, image_size, image_size))
    mask_start = int(image_size // 4)
    mask_end = int(image_size * 3 / 4)
    mask[:, :, mask_start:mask_end, mask_start:mask_end] = 0.
    return mask
    
  elif mask_type == "checkerboard":
    return get_checkerboard_mask(image_size=image_size, 
                                 num_blocks=checkerboard_num_blocks, 
                                 batch_size=batch_size, 
                                 num_channels=num_channels)
  elif mask_type == "checkerboard_alt":
    if maskgen_offset % 2 == 0:
      inverted = False
    else:
      inverted = True
    return get_checkerboard_mask(image_size=image_size,
                                 num_blocks=checkerboard_num_blocks,
                                 batch_size=batch_size,
                                 num_channels=num_channels, inverted=inverted)
    
  else:
    raise NotImplementedError
    

def get_checkerboard_mask(image_size, num_blocks, batch_size, num_channels=3, inverted=False):
  assert image_size % num_blocks == 0, 'image_size must be divisible by num_blocks'
  block_len = int(image_size // num_blocks)
  mask = torch.ones((batch_size, num_channels, num_blocks, num_blocks))
  if inverted:
    mask[:, :, ::2, ::2] = 0.
    mask[:, :, 1::2, 1::2] = 0.
  else:
    mask[:,:,1::2, ::2] = 0.
    mask[:,:,::2, 1::2] = 0.
  return mask.repeat_interleave(repeats=block_len,dim=2).repeat_interleave(repeats=block_len,dim=3)

######### Adapted some mask generation code from SimMIM
# https://github.com/microsoft/SimMIM/blob/0e7174608f5105f3d573f67724008c89b23b7fa2/data/data_simmim.py
class MaskGenerator:
  def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
    self.input_size = input_size
    self.mask_patch_size = mask_patch_size
    self.model_patch_size = model_patch_size
    self.mask_ratio = mask_ratio
        
    assert self.input_size % self.mask_patch_size == 0
    assert self.mask_patch_size % self.model_patch_size == 0
        
    self.rand_size = self.input_size // self.mask_patch_size
    self.scale = self.mask_patch_size // self.model_patch_size
        
    self.token_count = self.rand_size ** 2
    self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
  def __call__(self):
    mask_idx = np.random.permutation(self.token_count)[:self.mask_count]

    mask = np.zeros(self.token_count, dtype=int)
    mask[mask_idx] = 1
        
    mask = mask.reshape((self.rand_size, self.rand_size))
    mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
    return mask