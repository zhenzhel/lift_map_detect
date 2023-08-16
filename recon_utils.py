import torch
import torch.nn as nn
import torch.nn.functional as F
from models import utils as mutils
import losses
import sde_lib
import tqdm
import functools
from sampling import shared_predictor_update_fn, shared_corrector_update_fn
from models.ema import ExponentialMovingAverage
from utils import restore_checkpoint
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import numpy as np
from skimage.metrics import structural_similarity as ssim
import lpips
from simclr.resnet import get_resnet, name_to_params

def read_file(filename):
  f = open(filename, "r")
  x = f.read()
  f.close()
  return x

def write_file(filename, x):
  f = open(filename, "w")
  f.write(x)
  f.close()
  return 

def get_model(config, ckpt_path):
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)

  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
  state = restore_checkpoint(ckpt_path, state, device=config.device)
  return state["model"]

def get_model_ema(config, ckpt_path):
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)

  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
  state = restore_checkpoint(ckpt_path, state, device=config.device)
  state['ema'].copy_to(score_model.parameters())
  return score_model

def get_sde_and_eps(config):
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")
  return sde, eps

def get_diffusion_process(sde, scaler, eps=1e-3, device='cuda'):

  def diffuse(x, target_idx):
    x = scaler(x)
    timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
    t = torch.zeros(x.shape[0]).fill_(timesteps[target_idx]).to(device)
    z = torch.randn_like(x)
    mean, std = sde.marginal_prob(x, t)
    perturbed_x = mean + std[:, None, None, None] * z
    return perturbed_x

  return diffuse

def get_denoising_process(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def denoise(model, x, target_idx): # start_t in [0, sde.N)
    """ The PC sampler funciton.

    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    with torch.no_grad():
      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

      for i in tqdm.tqdm(range(target_idx, sde.N)):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t
        x, x_mean = corrector_update_fn(x, vec_t, model=model)
        x, x_mean = predictor_update_fn(x, vec_t, model=model)

      return inverse_scaler(x_mean if denoise else x) #, sde.N * (n_steps + 1)

  return denoise

class ImageDataset(Dataset):
  def __init__(self, file_list, image_size, center_crop=False):
    self.file_list = file_list
    self.image_size = image_size
    self.center_crop = center_crop

  def __len__(self):
    return len(self.file_list)

  def __getitem__(self, index):
    img = Image.open(self.file_list[index])
    if img.mode != "RGB":
      img = img.convert("RGB")
    if self.center_crop:
      img = center_crop_arr(img, self.image_size)
    img = T.ToTensor()(img)
    return img

def center_crop_arr(pil_image, image_size):
  # We are not on a new enough PIL to support the `reducing_gap`
  # argument, which uses BOX downsampling at powers of two first.
  # Thus, we do it by hand to improve downsample quality.
  while min(*pil_image.size) >= 2 * image_size:
    pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

  scale = image_size / min(*pil_image.size)
  pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

  arr = np.array(pil_image)
  crop_y = (arr.shape[0] - image_size) // 2
  crop_x = (arr.shape[1] - image_size) // 2
  return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]



def ssim_criterion(x, recon_x):
  x = x.detach().cpu().numpy().transpose((0,2,3,1))
  recon_x = recon_x.detach().cpu().numpy().transpose((0,2,3,1))
  ssim_vals = []
  for i in range(x.shape[0]):
    ssim_vals.append(ssim(x[i], recon_x[i], data_range=1, channel_axis = 2, gaussian_weights=False, full=True)[0])
  return ssim_vals

## taken from https://github.com/eladrich/pixel2style2pixel/blob/334f45e43335e92822744849a546c82e1cc47bb3/criteria/moco_loss.py#L7
# stylegan inversion psp
class MocoLoss(nn.Module):

  def __init__(self, ckpt_path):
    super(MocoLoss, self).__init__()
    print("Loading MOCO model from path: {}".format(ckpt_path))
    self.ckpt_path = ckpt_path
    self.model = self.__load_model()
    self.model.cuda()
    self.model.eval()
    
  def __load_model(self):
    import torchvision.models as models
    model = models.__dict__["resnet50"]()
    # freeze all layers but the last fc
    for name, param in model.named_parameters():
      if name not in ['fc.weight', 'fc.bias']:
        param.requires_grad = False
    checkpoint = torch.load(self.ckpt_path, map_location="cpu")
    state_dict = checkpoint['state_dict']
    # rename moco pre-trained keys
    for k in list(state_dict.keys()):
      # retain only encoder_q up to before the embedding layer
      if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
        # remove prefix
        state_dict[k[len("module.encoder_q."):]] = state_dict[k]
      # delete renamed or unused k
      del state_dict[k]
    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    # remove output layer
    model = nn.Sequential(*list(model.children())[:-1]).cuda()
    return model

  def extract_feats(self, x, resize_fn=None):
    if resize_fn is not None:
      x = resize_fn(x)
    x_feats = self.model(x)
    x_feats = nn.functional.normalize(x_feats, dim=1)
    x_feats = x_feats.squeeze()
    return x_feats

  def forward(self, orig, recon, resize_fn=None):
    n_samples = orig.shape[0]
    x_feats = self.extract_feats(orig, resize_fn=resize_fn)
    y_feats = self.extract_feats(recon, resize_fn=resize_fn)
    y_feats = y_feats.detach()
    loss = []
   
    for i in range(n_samples):
      cur_loss = y_feats[i].dot(x_feats[i]).item()
      loss.append(cur_loss)
    return loss

class MocoLoss2(nn.Module):
  def __init__(self, ckpt_path):
    super(MocoLoss2, self).__init__()
    tmp_model = MocoLoss(ckpt_path).model
    all_layers = list(tmp_model.children())
    cur_model = [nn.Sequential(*all_layers[:4])]
    for i in range(4, len(all_layers)-1):
      cur_model.append(all_layers[i])
    self.model = nn.Sequential(*cur_model)
    self.model.cuda()
    self.model.eval()

    self.L = len(self.model)

  def extract_feats(self, x, resize_fn=None):
    if resize_fn is not None:
      x = resize_fn(x)
    x_feats = []
    out = x
    for i in range(self.L):
      with torch.no_grad():
        out = self.model[i](out)
      x_feats.append(out)
    return x_feats
  
  def forward(self, orig, recon, ret_per_layer=False, dropout_rate=None, lo=0, hi=5, resize_fn=None):
    orig_outs = self.extract_feats(orig, resize_fn=resize_fn)
    recon_outs = self.extract_feats(recon, resize_fn=resize_fn)
    orig_feats, recon_feats, feat_diffs = dict(), dict(), dict()
    for i in range(lo, hi):
      orig_feats[i] = lpips.normalize_tensor(orig_outs[i])
      recon_feats[i] = lpips.normalize_tensor(recon_outs[i])
      feat_diffs[i] = (orig_feats[i]-recon_feats[i])**2
    
    if dropout_rate is None:
      res = [lpips.spatial_average(
        feat_diffs[i].sum(dim=1,keepdim=True), keepdim=True) for i in range(lo,hi)]
    else:
      res = [lpips.spatial_average(
        F.dropout(feat_diffs[i].sum(dim=1,keepdim=True), p=dropout_rate), keepdim=True) for i in range(lo,hi)]
    
    val = 0
    for i in range(lo, hi):
      val += res[i-lo]
        
    if(ret_per_layer):
      return (val, res)
    else:
      return val.flatten().detach().cpu().numpy()

class SimCLRv2Loss(nn.Module):
  def __init__(self, ckpt_path):
    super(SimCLRv2Loss, self).__init__()

    self.ckpt_path = ckpt_path
    self.model, _ = get_resnet(*name_to_params(ckpt_path))
    self.model.load_state_dict(torch.load(ckpt_path)['resnet'])
    self.model.cuda()
    self.model.eval()
  
  def extract_feats(self, x, resize_fn=None, normalize=True):
    # resizing stuff
    if resize_fn is not None:
      x = resize_fn(x)

    x_feats = self.model(x, apply_fc=False)
    if normalize:
      x_feats = F.normalize(x_feats, dim=1)
    #x_feats = x_feats.squeeze()
    return x_feats
  
  def forward(self, orig, recon, resize_fn=None, normalize=True, metric="cosine"):
    n_samples = orig.shape[0]
    x_feats = self.extract_feats(orig, resize_fn=resize_fn, normalize=normalize).detach()
    y_feats = self.extract_feats(recon, resize_fn=resize_fn, normalize=normalize).detach()

    loss = []
    for i in range(n_samples):
      if metric == "cosine":
        cur_loss = y_feats[i].dot(x_feats[i]).item()
      elif metric == "l2":
        cur_loss = ((y_feats[i]-x_feats[i])**2).mean().item()
      else:
        raise NotImplementedError
      loss.append(cur_loss)

    return loss