import torch
import os
from torch import nn
from torch.nn import functional as F
from utils.torch import *
from utils.config import Config
from .common.mlp import MLP
from .common.dist import *
from . import model_lib


def pred_loss(data, cfg):
    diff = data['env_pred'] - data['env_parameter'].unsqueeze(0)
    dist = diff.pow(2)
    loss_unweighted = dist
    if cfg.get('normalize', True):
        loss_unweighted = loss_unweighted.mean()
    else:
        loss_unweighted = loss_unweighted.sum()
    loss = loss_unweighted * cfg['weight']
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
        self.train_w_mean = cfg.get('train_w_mean', False)
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
        self.qnet_mlp = cfg.get('qnet_mlp', [512, 256])
        self.q_mlp = MLP(self.pred_model_dim * 8, self.qnet_mlp, 'relu')
        self.q_mlp2 = MLP(self.q_mlp.out_dim * 2, [256], 'relu')
        self.q_b = nn.Linear(self.q_mlp.out_dim * 2, nz)

    def set_device(self, device):
        self.device = device
        self.to(device)
        self.pred_model[0].set_device(device)

    def set_data(self, data):
        self.pred_model[0].set_data(data)
        self.data = self.pred_model[0].data

    def main(self, mean=False, need_weights=False):
        pred_model = self.pred_model[0]
        if hasattr(pred_model, 'use_map') and pred_model.use_map:
            self.data['map_enc'] = pred_model.map_encoder(self.data['agent_maps'])
        pred_model.context_encoder(self.data)
        target_latent = self.data['context_enc']
        enc_shape0 = target_latent.shape[0] // 8
        latent_vec_over_agt = target_latent.view(8, -1, 256).permute(1, 0, 2).reshape(-1, 8*256)
        #latent_vec = torch.max(latent_vec_over_agt, dim=0, keepdim=True).values
        qnet_h = self.q_mlp(latent_vec_over_agt)
        qnet_max = torch.max(qnet_h, dim=0, keepdim=True)[0]
        qnet_mean = torch.mean(qnet_h, dim=0, keepdim=True)
        qnet_h = torch.cat([qnet_mean, qnet_max], dim=1)
        b = self.q_b(qnet_h).view(-1, self.nz)

        self.data['env_pred'] = b

        return self.data
    
    def forward(self):
        return self.main(mean=self.train_w_mean)

    def inference(self, mode, sample_num, need_weights=False):
        self.main(mean=True, need_weights=need_weights)
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