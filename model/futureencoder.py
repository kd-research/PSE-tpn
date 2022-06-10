import torch
from torch import nn

from model.agentformer_mask import generate_mask
from model.positionalagentenc import PositionalAgentEncoding
from model.agentformer_lib import AgentFormerDecoderLayer, AgentFormerDecoder
from model.common.dist import Normal, Categorical
from model.common.mlp import MLP
from utils.torch import rotation_2d_torch
from utils.utils import initialize_weights


""" Future Encoder """


class FutureEncoder(nn.Module):
    def __init__(self, cfg, ctx, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.context_dim = context_dim = ctx['context_dim']
        self.forecast_dim = forecast_dim = ctx['forecast_dim']
        self.nz = ctx['nz']
        self.z_type = ctx['z_type']
        self.z_tau_annealer = ctx.get('z_tau_annealer', None)
        self.model_dim = ctx['tf_model_dim']
        self.ff_dim = ctx['tf_ff_dim']
        self.nhead = ctx['tf_nhead']
        self.dropout = ctx['tf_dropout']
        self.nlayer = cfg.get('nlayer', 6)
        self.out_mlp_dim = cfg.get('out_mlp_dim', None)
        self.input_type = ctx['fut_input_type']
        self.pooling = cfg.get('pooling', 'mean')
        self.agent_enc_shuffle = ctx['agent_enc_shuffle']
        self.vel_heading = ctx['vel_heading']
        # networks
        in_dim = forecast_dim * len(self.input_type)
        if 'map' in self.input_type:
            in_dim += ctx['map_enc_dim'] - forecast_dim
        self.input_fc = nn.Linear(in_dim, self.model_dim)

        decoder_layers = AgentFormerDecoderLayer(ctx['tf_cfg'], self.model_dim, self.nhead, self.ff_dim, self.dropout)
        self.tf_decoder = AgentFormerDecoder(decoder_layers, self.nlayer)

        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout, concat=ctx['pos_concat'], max_a_len=ctx['max_agent_len'], use_agent_enc=ctx['use_agent_enc'], agent_enc_learn=ctx['agent_enc_learn'])
        num_dist_params = 2 * self.nz if self.z_type == 'gaussian' else self.nz     # either gaussian or discrete
        if self.out_mlp_dim is None:
            self.q_z_net = nn.Linear(self.model_dim, num_dist_params)
        else:
            self.out_mlp = MLP(self.model_dim, self.out_mlp_dim, 'relu')
            self.q_z_net = nn.Linear(self.out_mlp.out_dim, num_dist_params)
        # initialize
        initialize_weights(self.q_z_net.modules())

    def forward(self, data, reparam=True):
        traj_in = []
        for key in self.input_type:
            if key == 'pos':
                traj_in.append(data['fut_motion'])
            elif key == 'vel':
                vel = data['fut_vel']
                if self.vel_heading:
                    vel = rotation_2d_torch(vel, -data['heading'])[0]
                traj_in.append(vel)
            elif key == 'norm':
                traj_in.append(data['fut_motion_norm'])
            elif key == 'scene_norm':
                traj_in.append(data['fut_motion_scene_norm'])
            elif key == 'heading':
                hv = data['heading_vec'].unsqueeze(0).repeat((data['fut_motion'].shape[0], 1, 1))
                traj_in.append(hv)
            elif key == 'map':
                map_enc = data['map_enc'].unsqueeze(0).repeat((data['fut_motion'].shape[0], 1, 1))
                traj_in.append(map_enc)
            else:
                raise ValueError('unknown input_type!')
        traj_in = torch.cat(traj_in, dim=-1)
        tf_in = self.input_fc(traj_in.view(-1, traj_in.shape[-1])).view(-1, 1, self.model_dim)
        agent_enc_shuffle = data['agent_enc_shuffle'] if self.agent_enc_shuffle else None
        tf_in_pos = self.pos_encoder(tf_in, num_a=data['agent_num'], agent_enc_shuffle=agent_enc_shuffle)

        mem_agent_mask = data['agent_mask'].clone()
        tgt_agent_mask = data['agent_mask'].clone()
        mem_mask = generate_mask(tf_in.shape[0], data['context_enc'].shape[0], data['agent_num'], mem_agent_mask).to(tf_in.device)
        tgt_mask = generate_mask(tf_in.shape[0], tf_in.shape[0], data['agent_num'], tgt_agent_mask).to(tf_in.device)

        tf_out, _ = self.tf_decoder(tf_in_pos, data['context_enc'], memory_mask=mem_mask, tgt_mask=tgt_mask, num_agent=data['agent_num'])
        tf_out = tf_out.view(traj_in.shape[0], -1, self.model_dim)

        if self.pooling == 'mean':
            h = torch.mean(tf_out, dim=0)
        else:
            h = torch.max(tf_out, dim=0)[0]
        if self.out_mlp_dim is not None:
            h = self.out_mlp(h)
        q_z_params = self.q_z_net(h)
        if self.z_type == 'gaussian':
            data['q_z_dist'] = Normal(params=q_z_params)
        else:
            data['q_z_dist'] = Categorical(logits=q_z_params, temp=self.z_tau_annealer.val())
        data['q_z_samp'] = data['q_z_dist'].rsample()
