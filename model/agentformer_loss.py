import torch
import numpy


def compute_motion_mse(data, cfg):
    diff = data['fut_motion_orig'] - data['train_dec_motion']
    if cfg.get('mask', True):
        mask = data['fut_mask']
        diff *= mask.unsqueeze(2)
    loss_unweighted = diff.pow(2).sum() 
    if cfg.get('normalize', True):
        loss_unweighted /= diff.shape[0]
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def compute_z_kld(data, cfg):
    loss_unweighted = data['q_z_dist'].kl(data['p_z_dist']).sum()
    if cfg.get('normalize', True):
        loss_unweighted /= data['batch_size']
    loss_unweighted = loss_unweighted.clamp_min_(cfg.min_clip)
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def compute_sample_loss(data, cfg):
    diff = data['infer_dec_motion'] - data['fut_motion_orig'].unsqueeze(1)
    if cfg.get('mask', True):
        mask = data['fut_mask'].unsqueeze(1).unsqueeze(-1)
        diff *= mask
    dist = diff.pow(2).sum(dim=-1).sum(dim=-1)
    loss_unweighted = dist.min(dim=1)[0]
    if cfg.get('normalize', True):
        loss_unweighted = loss_unweighted.mean()
    else:
        loss_unweighted = loss_unweighted.sum()
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


# noinspection PyTypeChecker
def compute_z_prior(data, cfg) -> torch.Tensor:
    # data q_z_samp
    latent_size = data['q_z_samp'].shape[1]
    z = data['q_z_samp']
    loc = torch.zeros((latent_size,)).cuda()
    var = torch.ones((latent_size,)).cuda()

    elementary_loss = -0.5 * torch.log(2 * numpy.pi * var) - torch.pow(z - loc, 2) / (2 * var)
    loss_unweighted = torch.clip(elementary_loss.mean((-1,)), -3, 0)

    # data context_enc
    latent_size = data['context_enc'].shape[1]
    z = data['context_enc']
    loc = torch.zeros((latent_size,)).cuda()
    var = torch.ones((latent_size,)).cuda()

    elementary_loss = -0.5 * torch.log(2 * numpy.pi * var) - torch.pow(z - loc, 2) / (2 * var)
    loss_unweighted1 = torch.clip(elementary_loss.mean((-1,)), -3, 0)

    if cfg.get('normalize', True):
        loss_unweighted = (loss_unweighted.mean() + loss_unweighted1.mean())/2
    else:
        loss_unweighted = loss_unweighted.sum() + loss_unweighted1.sum()
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted

def compute_y_dist(data, cfg):
    loss_unweighted = (data["train_seq_out"] - data["train_dist_seq_out"]).abs().sum(dim=-1)
    if cfg.get('normalize', True):
        loss_unweighted = loss_unweighted.mean()
    else:
        loss_unweighted = loss_unweighted.sum()
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted

loss_func = {
    'mse': compute_motion_mse,
    'kld': compute_z_kld,
    'sample': compute_sample_loss,
    'zpr': compute_z_prior,
    'ydist': compute_y_dist
}