import copy

import torch
from torch import nn

from model.agentformer_mask import generate_ar_mask, generate_mask
from model.positionalagentenc import PositionalAgentEncoding
from model.agentformer_lib import AgentFormerDecoderLayer, AgentFormerDecoder
from model.common.dist import Normal, Categorical
from model.common.mlp import MLP
from utils.torch import rotation_2d_torch
from utils.utils import initialize_weights

""" Future Decoder """


class FutureDecoder(nn.Module):
    def __init__(self, cfg, ctx, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.ar_detach = ctx['ar_detach']
        self.context_dim = context_dim = ctx['context_dim']
        self.forecast_dim = forecast_dim = ctx['forecast_dim']
        self.pred_scale = cfg.get('pred_scale', 1.0)
        self.pred_type = ctx['pred_type']
        self.sn_out_type = ctx['sn_out_type']
        self.sn_out_heading = ctx['sn_out_heading']
        self.input_type = ctx['dec_input_type']
        self.future_frames = ctx['future_frames']
        self.past_frames = ctx['past_frames']
        self.nz = ctx['nz']
        self.z_type = ctx['z_type']
        self.model_dim = ctx['tf_model_dim']
        self.ff_dim = ctx['tf_ff_dim']
        self.nhead = ctx['tf_nhead']
        self.dropout = ctx['tf_dropout']
        self.nlayer = cfg.get('nlayer', 6)
        self.out_mlp_dim = cfg.get('out_mlp_dim', None)
        self.pos_offset = cfg.get('pos_offset', False)
        self.agent_enc_shuffle = ctx['agent_enc_shuffle']
        self.learn_prior = ctx['learn_prior']
        # networks
        in_dim = forecast_dim + len(self.input_type) * forecast_dim + self.nz
        if 'map' in self.input_type:
            in_dim += ctx['map_enc_dim'] - forecast_dim
        self.input_fc = nn.Linear(in_dim, self.model_dim)

        decoder_layers = AgentFormerDecoderLayer(ctx['tf_cfg'], self.model_dim, self.nhead, self.ff_dim, self.dropout)
        self.tf_decoder = AgentFormerDecoder(decoder_layers, self.nlayer)

        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout, concat=ctx['pos_concat'],
                                                   max_a_len=ctx['max_agent_len'], use_agent_enc=ctx['use_agent_enc'],
                                                   agent_enc_learn=ctx['agent_enc_learn'])
        if self.out_mlp_dim is None:
            self.out_fc = nn.Linear(self.model_dim, forecast_dim)
        else:
            in_dim = self.model_dim
            self.out_mlp = MLP(in_dim, self.out_mlp_dim, 'relu')
            self.out_fc = nn.Linear(self.out_mlp.out_dim, forecast_dim)
        initialize_weights(self.out_fc.modules())
        if self.learn_prior:
            num_dist_params = 2 * self.nz if self.z_type == 'gaussian' else self.nz  # either gaussian or discrete
            self.p_z_net = nn.Linear(self.model_dim, num_dist_params)
            initialize_weights(self.p_z_net.modules())

    def decode_traj_ar(self, data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num,
                       need_weights=False):
        agent_num = data['agent_num']
        if self.pred_type == 'vel':
            dec_in = pre_vel[[-1]]
        elif self.pred_type == 'pos':
            dec_in = pre_motion[[-1]]
        elif self.pred_type == 'scene_norm':
            dec_in = pre_motion_scene_norm[[-1]]
        else:
            dec_in = torch.zeros_like(pre_motion[[-1]])
        dec_in = dec_in.view(-1, sample_num, dec_in.shape[-1])
        z_in = z.view(-1, sample_num, z.shape[-1])
        in_arr = [dec_in, z_in]
        for key in self.input_type:
            if key == 'heading':
                heading = data['heading_vec'].unsqueeze(1).repeat((1, sample_num, 1))
                in_arr.append(heading)
            elif key == 'map':
                map_enc = data['map_enc'].unsqueeze(1).repeat((1, sample_num, 1))
                in_arr.append(map_enc)
            else:
                raise ValueError('wrong decode input type!')
        dec_in_z = torch.cat(in_arr, dim=-1)

        mem_agent_mask = data['agent_mask'].clone()
        tgt_agent_mask = data['agent_mask'].clone()

        for i in range(self.future_frames):
            tf_in = self.input_fc(dec_in_z.view(-1, dec_in_z.shape[-1])).view(dec_in_z.shape[0], -1, self.model_dim)
            agent_enc_shuffle = data['agent_enc_shuffle'] if self.agent_enc_shuffle else None
            tf_in_pos = self.pos_encoder(tf_in, num_a=agent_num, agent_enc_shuffle=agent_enc_shuffle,
                                         t_offset=self.past_frames - 1 if self.pos_offset else 0)
            # tf_in_pos = tf_in
            mem_mask = generate_mask(tf_in.shape[0], context.shape[0], data['agent_num'], mem_agent_mask).to(
                tf_in.device)
            tgt_mask = generate_ar_mask(tf_in_pos.shape[0], agent_num, tgt_agent_mask).to(tf_in.device)

            tf_out, attn_weights = self.tf_decoder(tf_in_pos, context, memory_mask=mem_mask, tgt_mask=tgt_mask,
                                                   num_agent=data['agent_num'], need_weights=need_weights)

            out_tmp = tf_out.view(-1, tf_out.shape[-1])
            if self.out_mlp_dim is not None:
                out_tmp = self.out_mlp(out_tmp)
            seq_out = self.out_fc(out_tmp).view(tf_out.shape[0], -1, self.forecast_dim)
            if self.pred_type == 'scene_norm' and self.sn_out_type in {'vel', 'norm'}:
                norm_motion = seq_out.view(-1, agent_num * sample_num, seq_out.shape[-1])
                if self.sn_out_type == 'vel':
                    norm_motion = torch.cumsum(norm_motion, dim=0)
                if self.sn_out_heading:
                    angles = data['heading'].repeat_interleave(sample_num)
                    norm_motion = rotation_2d_torch(norm_motion, angles)[0]
                seq_out = norm_motion + pre_motion_scene_norm[[-1]]
                seq_out = seq_out.view(tf_out.shape[0], -1, seq_out.shape[-1])
            if self.ar_detach:
                out_in = seq_out[-agent_num:].clone().detach()
            else:
                out_in = seq_out[-agent_num:]
            # create dec_in_z
            in_arr = [out_in, z_in]
            for key in self.input_type:
                if key == 'heading':
                    in_arr.append(heading)
                elif key == 'map':
                    in_arr.append(map_enc)
                else:
                    raise ValueError('wrong decoder input type!')
            out_in_z = torch.cat(in_arr, dim=-1)
            dec_in_z = torch.cat([dec_in_z, out_in_z], dim=0)

        seq_out = seq_out.view(-1, agent_num * sample_num, seq_out.shape[-1])
        data[f'{mode}_seq_out'] = seq_out

        if self.pred_type == 'vel':
            dec_motion = torch.cumsum(seq_out, dim=0)
            dec_motion += pre_motion[[-1]]
        elif self.pred_type == 'pos':
            dec_motion = seq_out.clone()
        elif self.pred_type == 'scene_norm':
            dec_motion = seq_out + data['scene_orig']
        else:
            dec_motion = seq_out + pre_motion[[-1]]

        dec_motion = dec_motion.transpose(0, 1).contiguous()  # M x frames x 7
        if mode == 'infer':
            dec_motion = dec_motion.view(-1, sample_num, *dec_motion.shape[1:])  # M x Samples x frames x 3
        data[f'{mode}_dec_motion'] = dec_motion
        if need_weights:
            data['attn_weights'] = attn_weights

    def decode_traj_dist(self, data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num):
        DIST_SEQ_OUT = f"{mode}_seq_out"
        DIST_DEC_MOTION = f"{mode}_dec_motion"

        data_dup = copy.copy(data)
        z_dist = z + torch.randn_like(z)
        ce = data_dup["context_enc"]
        data_dup["context_enc"] = ce + torch.randn_like(ce)
        self.decode_traj_ar(data_dup, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z_dist, sample_num)

        data[DIST_SEQ_OUT] = data_dup[DIST_SEQ_OUT]
        data[DIST_DEC_MOTION] = data_dup[DIST_DEC_MOTION]

    def decode_traj_batch(self, data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num):
        raise NotImplementedError

    def forward(self, data, mode, sample_num=1, autoregress=True, z=None, need_weights=False):
        context = data['context_enc'].repeat_interleave(sample_num, dim=1)  # 80 x 64
        pre_motion = data['pre_motion'].repeat_interleave(sample_num, dim=1)  # 10 x 80 x 2
        pre_vel = data['pre_vel'].repeat_interleave(sample_num, dim=1) if self.pred_type == 'vel' else None
        pre_motion_scene_norm = data['pre_motion_scene_norm'].repeat_interleave(sample_num, dim=1)

        # p(z)
        prior_key = 'p_z_dist' + ('_infer' if mode == 'infer' else '')
        if self.learn_prior:
            h = data['agent_context'].repeat_interleave(sample_num, dim=0)
            p_z_params = self.p_z_net(h)
            if self.z_type == 'gaussian':
                data[prior_key] = Normal(params=p_z_params)
            else:
                data[prior_key] = Categorical(params=p_z_params)
        else:
            if self.z_type == 'gaussian':
                data[prior_key] = Normal(mu=torch.zeros(pre_motion.shape[1], self.nz).to(pre_motion.device),
                                         logvar=torch.zeros(pre_motion.shape[1], self.nz).to(pre_motion.device))
            else:
                data[prior_key] = Categorical(logits=torch.zeros(pre_motion.shape[1], self.nz).to(pre_motion.device))

        if z is None:
            if mode in {'train', 'recon'}:
                z = data['q_z_samp'] if mode == 'train' else data['q_z_dist'].mode()
            elif mode == 'infer':
                z = data['p_z_dist_infer'].sample()
            else:
                raise ValueError('Unknown Mode!')

        if autoregress:
            self.decode_traj_ar(data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num,
                                need_weights=need_weights)
            if mode == 'train':
                self.decode_traj_dist(data, f"{mode}_dist", context, pre_motion, pre_vel, pre_motion_scene_norm, z,
                                      sample_num)
        else:
            self.decode_traj_batch(data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num)
