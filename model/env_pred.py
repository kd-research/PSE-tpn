import torch
import os
from torch import nn
from torch.nn import functional as F
from utils.torch import *
from utils.config import Config
from .common.mlp import MLP
from .common.dist import *
from . import model_lib
from . import diversity_weight
from torch.nn.init import xavier_uniform_


def pred_loss(data, cfg):
    diff = data['env_pred'] - data['env_parameter'].unsqueeze(0)
    dist = diff.pow(2)
    loss_unweighted = dist
    if cfg.get('normalize', True):
        loss_unweighted = loss_unweighted.mean()
    else:
        loss_unweighted = loss_unweighted.sum()
    loss = loss_unweighted * cfg['weight']
    # loss = loss_unweighted * cfg['weight'] * diversity_weight.diversity_weight(data['context_enc'].view(-1, 8, 256).mean(0))
    accuracy = (torch.abs(data['env_pred'] - data['env_parameter']).detach().cpu().numpy() < 1).sum() / data['env_parameter'].shape[0]
    print(f"\r{accuracy} ", end="")
    return loss, loss_unweighted


loss_func = {
    'pred': pred_loss,
}


class EnvPred(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device('cpu')
        self.cfg = cfg
        self.nz = nz = cfg.nz
        self.loss_cfg = self.cfg.loss_cfg
        self.loss_names = list(self.loss_cfg.keys())

        pred_cfg = Config(cfg.pred_cfg, tmp=False, create_dirs=False)
        pred_model = model_lib.model_dict[pred_cfg.model_id](pred_cfg)
        self.pred_model_dim = pred_cfg.tf_model_dim
        if cfg.pred_epoch > 0:
            cp_path = pred_cfg.model_path % cfg.pred_epoch
            print('loading model from checkpoint: %s' % cp_path)
            model_cp = torch.load(cp_path, map_location='cpu')
            pred_model.load_state_dict(model_cp['model_dict'])
        pred_model.eval()
        self.pred_model = [pred_model]

        # Dlow's Q net
        self.q_mlp = nn.Sequential(nn.Linear(4500, 900), nn.ReLU(), nn.Linear(900, 50), nn.ReLU())
        self.q_b = nn.Sequential(nn.Linear(self.pred_model_dim * 50, 256), nn.ReLU(), nn.Linear(256, nz))
        self.tanh = nn.Tanh()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.q_mlp.apply(init_weights)
        self.q_b.apply(init_weights)

    def set_device(self, device):
        self.device = device
        self.to(device)
        self.pred_model[0].set_device(device)

    def set_data(self, data):
        self.pred_model[0].set_data(data)
        self.data = self.pred_model[0].data

    def main(self, random_latent=False, need_z=False):
        pred_model = self.pred_model[0]
        if hasattr(pred_model, 'use_map') and pred_model.use_map:
            self.data['map_enc'] = pred_model.map_encoder(self.data['agent_maps'])
        pred_model.context_encoder(self.data)
        if need_z:
            pred_model.future_encoder(self.data)
        target_latent = self.data['context_enc']
        lsize = target_latent.shape[-1]
        padded = torch.zeros((lsize, 4500), device=self.device)
        pad_len = min(target_latent.shape[0], 4500)
        padded[:, 0:pad_len] = target_latent[0:pad_len, 0, :].T
        if random_latent:
            padded = torch.randn_like(padded)
        qnet_h = self.q_mlp(padded).ravel()
        qb = self.q_b(qnet_h).view(-1, self.nz)
        b = self.tanh(qb)
        self.data['env_pred'] = b

        return self.data

    def forward(self):
        return self.main()

    def inference(self, *args, random_latent=False, **kwargs):
        self.main(random_latent=random_latent, need_z=True)
        res = self.data['env_pred']
        return res, self.data

    def compute_loss(self):
        total_loss = 0
        loss_dict = {}
        loss_unweighted_dict = {}
        for loss_name in self.loss_names:
            loss, loss_unweighted = loss_func[loss_name](self.data, self.loss_cfg[loss_name])
            total_loss += loss
            loss_dict[loss_name] = loss.item()
            loss_unweighted_dict[loss_name] = loss_unweighted.item()
        return total_loss, loss_dict, loss_unweighted_dict

    def step_annealer(self):
        pass
