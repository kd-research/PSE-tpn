import math
import logging
import numpy as np

from .steersim import steersimProcess

logger = logging.Logger(__name__)

class SteersimSegmentedProcess(steersimProcess):
    def __init__(self, *args, **kwargs):
        # self.min_past_frames is meaningless in this context
        # self.min_future_frame is used to validate agents in each sample
        super().__init__(*args, **kwargs)
        if np.isnan(self.gt).any():
            raise ValueError('gt contains NaN')

    def get_total_available_sample_size(self):
        assert self.frame_skip == 1, "frame skip is not considered yet"
        # trajectories will be splited in multiple segments
        # based on the past/future frame size
        target_frame_size = self.past_frames + self.future_frames
        full_frame_size = self.get_max_total_frame()
        full_sample_num, rest_frame = divmod(full_frame_size, target_frame_size)

        # last segment must have at least past_frames + min_future_frames
        # otherwise this segment will be discarded
        if rest_frame >= self.past_frames + self.min_future_frames:
            full_sample_num += 1

        return full_sample_num

    def get_min_total_frame(self):
        # length of the fastest agent achieves the goal
        agent_ids = set(self.gt[:, 1])
        frame_by_id = []
        for aid in agent_ids:
            agent_frame = self.gt[self.gt[:, 1] == aid]
            frame_by_id.append(np.max(agent_frame) + 1)

        return int(min(frame_by_id))

    def get_max_total_frame(self):
        # length of the slowest agent achieves the goal
        agent_ids = set(self.gt[:, 1])
        frame_by_id = []
        for aid in agent_ids:
            agent_frame = self.gt[self.gt[:, 1] == aid]
            frame_by_id.append(np.max(agent_frame[:, 0]) + 1)

        return int(max(frame_by_id))


    def __call__(self, sample_index, *args, **kwargs):
        assert self.frame_skip == 1, "frame skip is not considered yet"

        # Steersim.call(8) have past_frame [1..8] and future frame (9...)

        target_frame_size = self.past_frames + self.future_frames
        selected_split_frame_num = sample_index * target_frame_size + self.past_frames - 1

        pre_data = self.PreData(selected_split_frame_num)
        fut_data = self.FutureData(selected_split_frame_num)
        valid_id = self.get_valid_id(pre_data, fut_data)
        if len(valid_id) == 0:
            assert False, f"Unexpected empty sample: {self.seq_name} {sample_index} {selected_split_frame_num}"

        pred_mask = None
        heading = None

        pre_motion_3D, pre_motion_mask = self.PreMotion(pre_data, valid_id)
        fut_motion_3D, fut_motion_mask = self.FutureMotion(fut_data, valid_id)

        data = {'pre_motion_3D': pre_motion_3D, 'fut_motion_3D': fut_motion_3D, 'fut_motion_mask': fut_motion_mask,
                'pre_motion_mask': pre_motion_mask, 'pre_data': pre_data, 'fut_data': fut_data, 'heading': heading,
                'valid_id': valid_id, 'traj_scale': self.traj_scale, 'pred_mask': pred_mask,
                'scene_map': self.geom_scene_map, 'seq': self.seq_name, 'frame': selected_split_frame_num, "env_parameter": self.parm}

        return data
