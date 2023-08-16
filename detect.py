import argparse
import lpips
from sklearn.metrics import roc_auc_score
import pickle
from mask_utils import *
from recon_utils import *
import glob

parser = argparse.ArgumentParser(description='')
parser.add_argument('--result_path', default='', type=str, help='')
parser.add_argument('--metric', default='LPIPS', type=str, help='MSE, SSIM, LPIPS, SIMCLR')
parser.add_argument('--reps', default=1, type=int, help='')
parser.add_argument('--resize_type', default='bicubic', type=str, help='simclr resize type supports none, nearest, bicubic')
args = parser.parse_args()
print(vars(args))

reps = args.reps
root = args.result_path
metric = args.metric
num_batches_pos = len(glob.glob(root + 'pos/batch_*.pth'))
num_batches_neg = len(glob.glob(root + 'neg/batch_*.pth'))

print('number of positive chunks: {0}, negative chunks: {1}'.format(num_batches_pos, num_batches_neg))

all_pos = dict()
for r in tqdm.tqdm(range(reps)):
    pos = dict()

    pos0 = torch.load("{0}pos/batch_0.pth".format(root))
    pos_orig = pos0['orig']
    pos_recon = pos0['recon'][r]
    pos_masked = pos0['masked'][r]

    for i in range(1, num_batches_pos):
        pos_cur = torch.load("{0}pos/batch_{1}.pth".format(root, i))
        pos_orig = torch.cat([pos_orig, pos_cur['orig']], dim=0)
        pos_recon = torch.cat([pos_recon, pos_cur['recon'][r]], dim=0)
        pos_masked = torch.cat([pos_masked, pos_cur['masked'][r]], dim=0)

    pos['orig'] = pos_orig
    pos['recon'] = pos_recon
    pos['masked'] = pos_masked
    all_pos[r] = pos

all_neg = dict()
for r in tqdm.tqdm(range(reps)):
    neg = dict()

    neg0 = torch.load("{0}neg/batch_0.pth".format(root))
    neg_orig = neg0['orig']
    neg_recon = neg0['recon'][r]
    neg_masked = neg0['masked'][r]

    for i in range(1, num_batches_neg):
        neg_cur = torch.load("{0}neg/batch_{1}.pth".format(root, i))
        neg_orig = torch.cat([neg_orig, neg_cur['orig']], dim=0)
        neg_recon = torch.cat([neg_recon, neg_cur['recon'][r]], dim=0)
        neg_masked = torch.cat([neg_masked, neg_cur['masked'][r]], dim=0)

    neg['orig'] = neg_orig
    neg['recon'] = neg_recon
    neg['masked'] = neg_masked
    all_neg[r] = neg

# modification
n_data_pos = all_pos[0]['orig'].shape[0]
n_data_neg = all_neg[0]['orig'].shape[0]

all_pos_eval, all_neg_eval = np.zeros((reps, n_data_pos)), np.zeros((reps, n_data_neg))
if metric == 'SIMCLR':
    simclr_criterion = SimCLRv2Loss("pretrained/r50_1x_sk1.pth")
elif metric == 'LPIPS':
    lpips_criterion = lpips.LPIPS(net='alex', pretrained=True, lpips=True).cuda()
def simclr_resize(x):
    if args.resize_type == 'none':
        return x
    elif args.resize_type in ['nearest', 'bicubic']:
        result = F.interpolate(x, 224, mode=args.resize_type.split('_')[0])
        return result
    else:
        raise NotImplementedError


def lpips_scaler(x):
    return x * 2. - 1.

for r in tqdm.tqdm(range(reps)):
    print('round {0} results:'.format(r))
    pos = all_pos[r]
    neg = all_neg[r]
    if metric == 'MSE':
        all_pos_eval[r, :] = F.mse_loss(pos['recon'], pos['orig'], reduction="none").mean(
            dim=(1, 2, 3)).detach().cpu().numpy().reshape(-1)
        all_neg_eval[r, :] = F.mse_loss(neg['recon'], neg['orig'], reduction="none").mean(
            dim=(1, 2, 3)).detach().cpu().numpy().reshape(-1)
    elif metric == 'SSIM':
        all_pos_eval[r, :] = ssim_criterion(pos['orig'], pos['recon'])
        all_neg_eval[r, :] = ssim_criterion(neg['orig'], neg['recon'])
    elif metric == 'LPIPS':
        all_pos_eval[r,:] = lpips_criterion(lpips_scaler(pos['orig'].cuda()), lpips_scaler(pos['recon'].cuda())).flatten().detach().cpu().numpy()
        all_neg_eval[r,:] = lpips_criterion(lpips_scaler(neg['orig'].cuda()), lpips_scaler(neg['recon'].cuda())).flatten().detach().cpu().numpy()
    elif metric == 'SIMCLR':
        steps = 1000
        for lo in range(0, pos['orig'].shape[0], steps):
            hi = lo + steps
            with torch.no_grad():
                all_pos_eval[r, lo:hi] = simclr_criterion(pos['orig'][lo:hi].cuda(), pos['recon'][lo:hi].cuda(),
                                                            resize_fn=simclr_resize)

        for lo in range(0, neg['orig'].shape[0], steps):
            hi = lo + steps
            with torch.no_grad():
                all_neg_eval[r, lo:hi] = simclr_criterion(neg['orig'][lo:hi].cuda(), neg['recon'][lo:hi].cuda(),
                                                            resize_fn=simclr_resize)
    else:
        raise NotImplementedError

agg_fn = np.median
all_pos_s = agg_fn(all_pos_eval, axis=0)
all_neg_s = agg_fn(all_neg_eval, axis=0)

results = np.append(all_pos_s, all_neg_s)
labels = np.append(np.ones_like(all_pos_s), np.zeros_like(all_neg_s))
if metric == 'LPIPS' or metric == 'MSE':
    n1 = roc_auc_score(labels, results * (-1))
else:
    n1 = roc_auc_score(labels, results)
print("%s ROC AUC: %.4f" % (metric, n1))